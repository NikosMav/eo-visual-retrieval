import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eo_visual_retrieval.hashing import file_sha256
from eo_visual_retrieval.models import StacItemRecord
from eo_visual_retrieval.stac import (
    StacSearchConfig,
    _image_suffix,
    _safe_filename,
    materialize_previews,
    read_stac_jsonl,
    write_stac_jsonl,
)


def test_stac_config_rejects_unbounded_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        StacSearchConfig(
            api_url="https://example.test/stac",
            collection="example",
            bbox=(1.0, 2.0, 3.0, 4.0),
            datetime="2024-01-01/2024-01-31",
            limit=1001,
        )


def test_stac_manifest_contains_identity_but_no_hrefs(tmp_path: Path) -> None:
    record = StacItemRecord(
        api_url="https://example.test/stac",
        collection="sentinel",
        item_id="item-1",
        bbox=(1.0, 2.0, 3.0, 4.0),
        datetime="2024-01-01T00:00:00+00:00",
        asset_keys=("blue", "green", "red"),
        properties={"eo:cloud_cover": 5.0},
    )
    output = tmp_path / "stac.jsonl"
    write_stac_jsonl([record], output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["item_id"] == "item-1"
    assert "href" not in output.read_text(encoding="utf-8").lower()
    assert read_stac_jsonl(output) == [record]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("api_url", "http://example.test/stac", "HTTPS"),
        ("collection", "", "collection"),
        ("datetime", "", "datetime"),
        ("bbox", (3.0, 2.0, 1.0, 4.0), "longitudes"),
        ("bbox", (1.0, 4.0, 3.0, 2.0), "latitudes"),
        ("max_cloud_cover", 101.0, "max_cloud_cover"),
    ],
)
def test_stac_config_rejects_unsafe_or_impossible_queries(
    field: str, value: object, message: str
) -> None:
    arguments: dict[str, Any] = {
        "api_url": "https://example.test/stac",
        "collection": "example",
        "bbox": (1.0, 2.0, 3.0, 4.0),
        "datetime": "2024-01-01/2024-01-31",
        field: value,
    }

    with pytest.raises(ValueError, match=message):
        StacSearchConfig(**arguments)


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        ("image/png", ".png"),
        ("image/tiff; application=geotiff", ".tif"),
        ("image/jpeg", ".jpg"),
        ("image/webp", ".webp"),
    ],
)
def test_supported_asset_media_types_map_to_extensions(
    content_type: str, expected: str
) -> None:
    assert _image_suffix(content_type) == expected


@pytest.mark.parametrize("content_type", ["", "text/html", "application/json"])
def test_unsupported_asset_media_types_are_refused(content_type: str) -> None:
    with pytest.raises(ValueError, match="not a supported image"):
        _image_suffix(content_type)


@pytest.mark.parametrize(
    ("item_id", "expected"),
    [
        ("S2A_MSIL2A_20240625", "S2A_MSIL2A_20240625.tif"),
        ("../../etc/passwd", "etc-passwd.tif"),
        ("a/b\\c:d*e", "a-b-c-d-e.tif"),
        ("...", "item.tif"),
        ("", "item.tif"),
    ],
)
def test_asset_filenames_cannot_escape_the_output_directory(
    item_id: str, expected: str
) -> None:
    name = _safe_filename(item_id, ".tif")

    assert name.startswith(expected.removesuffix(".tif") + "-")
    assert name.endswith(".tif")
    assert name == _safe_filename(item_id, ".tif")
    assert "/" not in name and "\\" not in name


def test_stac_manifest_rejects_a_record_carrying_an_asset_url(tmp_path: Path) -> None:
    record = StacItemRecord(
        api_url="https://example.test/stac",
        collection="sentinel",
        item_id="item-1",
        bbox=None,
        datetime=None,
        asset_keys=("blue",),
        properties={"note": "see the href for details"},
    )

    with pytest.raises(ValueError, match="must not contain asset HREFs"):
        write_stac_jsonl([record], tmp_path / "stac.jsonl")


def test_empty_and_malformed_manifests_are_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="STAC manifest is empty"):
        read_stac_jsonl(empty)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"api_url": "https://example.test"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid STAC manifest record at line 1"):
        read_stac_jsonl(malformed)


@pytest.mark.parametrize("url", [
    "https://name:synthetic-password@example.test/stac",
    "https://example.test/stac?api_key=synthetic-value",
    "https://example.test/stac#synthetic-fragment",
    "https:///missing-host",
])
def test_access_bearing_api_urls_are_rejected_before_persistence(url: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="api_url"):
        StacSearchConfig(api_url=url, collection="test", bbox=(0, 0, 1, 1), datetime="2024-01-01")
    record: dict[str, Any] = {
        "api_url": url, "collection": "test", "item_id": "test",
        "bbox": None, "datetime": None, "asset_keys": [],
    }
    manifest = tmp_path / "unsafe.jsonl"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid STAC manifest"):
        read_stac_jsonl(manifest)


def test_preview_colliding_sanitized_names_keep_distinct_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests
    from pystac_client import Client

    records = [StacItemRecord(
        api_url="https://example.test/stac", collection="test", item_id=identity,
        bbox=None, datetime=None, asset_keys=("rendered_preview",),
    ) for identity in ("tile/a", "tile:a")]
    contents = {"https://example.test/a.png": b"first", "https://example.test/b.png": b"second"}
    items = dict(zip((row.item_id for row in records), contents, strict=True))
    collection = SimpleNamespace(get_item=lambda identity: SimpleNamespace(assets={
        "rendered_preview": SimpleNamespace(href=items[identity], media_type="image/png"),
    }))
    monkeypatch.setattr(Client, "open", lambda url: SimpleNamespace(
        get_collection=lambda name: collection,
    ))

    def response(url: str, **kwargs: Any) -> Any:
        return nullcontext(SimpleNamespace(
            headers={"content-type": "image/png"}, raise_for_status=lambda: None,
            iter_content=lambda chunk_size: iter([contents[url]]),
        ))

    monkeypatch.setattr(requests, "Session", lambda: SimpleNamespace(
        mount=lambda *args: None, get=response,
    ))
    images = materialize_previews(
        records, output_dir=tmp_path / "images", image_manifest=tmp_path / "images.jsonl",
    )
    assert images[0].path != images[1].path
    retained = [(tmp_path / "images" / row.path).read_bytes() for row in images]
    assert retained == [b"first", b"second"]
    assert all(row.metadata["sha256"] == file_sha256(tmp_path / "images" / row.path)
               for row in images)
    assert _safe_filename("tile/a", ".png", namespace="first") != _safe_filename(
        "tile/a", ".png", namespace="second"
    )


def test_object_storage_assets_explain_the_credential_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copernicus answers searches anonymously but stores pixels behind credentials.

    A generic "must use HTTPS" rejection would read as a bug in this project
    rather than as the provider's access model, so the message names it.
    """

    from pystac_client import Client

    record = StacItemRecord(
        api_url="https://stac.dataspace.copernicus.eu/v1",
        collection="sentinel-2-l2a",
        item_id="S2B_MSIL2A_20250830T092029_N0511_R093_T35SKC_20250830T113855",
        bbox=None,
        datetime=None,
        asset_keys=("B02_10m",),
    )
    collection = SimpleNamespace(get_item=lambda identity: SimpleNamespace(assets={
        "B02_10m": SimpleNamespace(
            href="s3://eodata/Sentinel-2/MSI/L2A/2025/08/30/scene.SAFE/B02_10m.jp2",
            media_type="image/jp2",
        ),
    }))
    monkeypatch.setattr(Client, "open", lambda url: SimpleNamespace(
        get_collection=lambda name: collection,
    ))

    with pytest.raises(ValueError, match="object storage, not HTTPS"):
        materialize_previews(
            [record],
            output_dir=tmp_path / "images",
            image_manifest=tmp_path / "images.jsonl",
            asset_key="B02_10m",
        )


def test_allowlist_covers_both_generations_of_extension_names() -> None:
    """One provider's tile identity is another's, under a different key."""

    from eo_visual_retrieval.stac import SAFE_PROPERTY_KEYS

    # Planetary Computer names, and the Copernicus equivalents.
    assert {"s2:mgrs_tile", "s2:processing_baseline"} <= set(SAFE_PROPERTY_KEYS)
    assert {"grid:code", "processing:version", "product:type"} <= set(SAFE_PROPERTY_KEYS)
    # Viewing geometry explains why one place looks different between passes.
    assert {"view:sun_elevation", "sat:relative_orbit"} <= set(SAFE_PROPERTY_KEYS)
    # Nothing credential- or infrastructure-shaped may enter a persisted manifest.
    assert not {"auth:schemes", "storage:schemes"} & set(SAFE_PROPERTY_KEYS)
