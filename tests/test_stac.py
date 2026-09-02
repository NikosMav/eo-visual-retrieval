import json
from pathlib import Path
from typing import Any

import pytest

from eo_visual_retrieval.models import StacItemRecord
from eo_visual_retrieval.stac import (
    StacSearchConfig,
    _image_suffix,
    _safe_filename,
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

    assert name == expected
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
