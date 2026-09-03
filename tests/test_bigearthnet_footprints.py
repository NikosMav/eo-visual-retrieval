"""Source identity, bounded streaming, geometry and atomic-failure checks."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eo_visual_retrieval.datasets import bigearthnet_footprints as footprints
from eo_visual_retrieval.datasets.bigearthnet import MetadataAsset
from eo_visual_retrieval.hashing import file_md5


def _identity(column: int = 26) -> str:
    return f"S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_{column:02d}_57"


def _tiff(**changes: Any) -> bytes:
    rio = pytest.importorskip("rasterio")
    options = dict(driver="GTiff", width=120, height=120, count=1, dtype="uint16",
                   crs="EPSG:32633", transform=rio.Affine(10, 0, 331200, 0, -10, 5331600))
    options.update(changes)
    if isinstance(options["transform"], tuple):
        options["transform"] = rio.Affine(*options["transform"])
    with rio.MemoryFile() as memory:
        with memory.open(**options) as source:
            source.write(np.zeros((options["height"], options["width"]), dtype=options["dtype"]), 1)
        return bytes(memory.read())


def _name(identity: str) -> str:
    return f"Reference_Maps/{identity.rsplit('_', 2)[0]}/{identity}/{identity}_reference_map.tif"


def _archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entries: list[tuple[str, bytes, bytes]],
) -> Path:
    zstd = pytest.importorskip("zstandard")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, payload, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
    path = tmp_path / footprints.REFERENCE_ARCHIVE_FILENAME
    path.write_bytes(zstd.ZstdCompressor().compress(buffer.getvalue()))
    monkeypatch.setattr(footprints, "REFERENCE_ARCHIVE_BYTES", path.stat().st_size)
    monkeypatch.setattr(footprints, "REFERENCE_ARCHIVE_MD5", file_md5(path))
    return path


def test_selected_headers_are_read_and_full_membership_is_still_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = {_identity(), _identity(27)}
    path = _archive(tmp_path, monkeypatch, [
        (_name(_identity()), _tiff(), tarfile.REGTYPE),
        (_name(_identity(27)), b"unparsed reference pixels", tarfile.REGTYPE),
    ])
    rows = list(footprints.reference_footprints(path, ids, selected_ids={_identity()}))
    assert len(rows) == 1
    assert rows[0]["left"] == 331200
    footprints.add_centers(rows)
    assert 12 < rows[0]["longitude"] < 15
    assert rows[0]["spatial_group"].startswith("epsg6933:")
    with pytest.raises(ValueError, match="absent from metadata"):
        list(footprints.reference_footprints(path, ids, selected_ids={"unknown"}))
    with pytest.raises(ValueError, match="missing 1 metadata IDs"):
        list(footprints.reference_footprints(path, ids | {_identity(28)}, selected_ids=set()))


@pytest.mark.parametrize("case, message", [
    ("duplicate", "duplicate or unknown"), ("foreign", "duplicate or unknown"),
    ("traversal", "unsafe"), ("absolute", "unexpected"), ("link", "member type"),
    ("size", "byte limit"), ("mismatch", "does not match"), ("window", "window exceeds"),
])
def test_malformed_reference_archives_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, message: str,
) -> None:
    identity = _identity()
    entries = [(_name(identity), _tiff(), tarfile.REGTYPE)]
    if case == "duplicate":
        entries *= 2
    elif case == "foreign":
        entries = [(_name(_identity(27)), _tiff(), tarfile.REGTYPE)]
    elif case == "traversal":
        entries = [("Reference_Maps/../bad", b"x", tarfile.REGTYPE)]
    elif case == "absolute":
        entries = [("/Reference_Maps/bad", b"x", tarfile.REGTYPE)]
    elif case == "link":
        entries = [(_name(identity), b"", tarfile.SYMTYPE)]
    elif case == "size":
        entries = [(_name(identity), bytes(footprints.MAX_MAP_BYTES + 1), tarfile.REGTYPE)]
    elif case == "mismatch":
        entries = [(_name(identity).replace("_reference_map", "_bad"), b"x", tarfile.REGTYPE)]
    elif case == "window":
        monkeypatch.setattr(footprints, "MAX_WINDOW_BYTES", 1)
    path = _archive(tmp_path, monkeypatch, entries)
    with pytest.raises(ValueError, match=message):
        list(footprints.reference_footprints(path, {identity}))


@pytest.mark.parametrize("change, message", [
    ({"width": 60}, "dimensions"), ({"dtype": "uint8"}, "dtype"),
    ({"crs": "EPSG:32632"}, "CRS"), ({"crs": None}, "CRS"),
    ({"transform": (20, 0, 331200, 0, -10, 5331600)}, "transform"),
])
def test_unexpected_reference_geometry_is_rejected(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        footprints._map_geometry(_tiff(**change), _identity())
    with pytest.raises(ValueError):
        footprints._tag_geometry(_tiff(**change), _identity())


def test_tag_geometry_matches_independent_rasterio_reader() -> None:
    payload = _tiff()
    assert footprints._tag_geometry(payload, _identity()) == footprints._map_geometry(
        payload, _identity(),
    )


def test_inventory_promotes_only_complete_verified_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source"
    source.mkdir()
    _archive(source, monkeypatch, [(_name(_identity()), _tiff(), tarfile.REGTYPE)])
    metadata = source / "metadata.parquet"
    pq.write_table(pa.Table.from_pylist([{"patch_id": _identity()}]), metadata)
    monkeypatch.setattr(footprints, "METADATA_ASSETS", (
        MetadataAsset(metadata.name, file_md5(metadata), "synthetic"),
    ))
    output = tmp_path / "inventory.parquet"
    report = footprints.build_inventory(source, output)
    assert report["patches"] == 1
    assert report["s2_subset_footprints_verified"] is False
    assert pq.read_table(output)["patch_id"].to_pylist() == [_identity()]
    with pytest.raises(ValueError, match="already exists"):
        footprints.build_inventory(source, output)
    with pytest.raises(ValueError, match="outside"):
        footprints.build_inventory(source, source / "inventory.parquet")
    monkeypatch.setattr(footprints, "MAX_INVENTORY_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds byte limit"):
        footprints.build_inventory(source, tmp_path / "oversize.parquet")
    assert not (tmp_path / "oversize.parquet").exists()
    assert not list(tmp_path.glob("*.tmp"))


class _Response(io.BytesIO):
    status = 200
    headers: dict[str, str] = {}


@pytest.mark.parametrize("case", ["valid", "oversize", "short", "corrupt", "status", "length"])
def test_download_budget_integrity_and_failure_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = _archive(source, monkeypatch, [(_name(_identity()), _tiff(), tarfile.REGTYPE)])
    payload = path.read_bytes()
    if case == "oversize":
        payload += b"x"
    elif case == "short":
        payload = payload[:-1]
    elif case == "corrupt":
        payload = b"x" + payload[1:]
    response = _Response(payload)
    if case == "status":
        response.status = 206
    if case == "length":
        response.headers = {"Content-Length": "1"}
    monkeypatch.setattr(footprints, "urlopen", lambda *args, **kwargs: response)
    target = tmp_path / "download"
    if case == "valid":
        destination = footprints.download_reference_archive(target)
        assert destination.read_bytes() == payload
        assert footprints.download_reference_archive(target) == destination
        destination.write_bytes(b"bad cache")
        with pytest.raises(ValueError, match="byte count"):
            footprints.download_reference_archive(target)
    else:
        with pytest.raises(ValueError):
            footprints.download_reference_archive(target)
        assert not list(target.iterdir())
