"""Exercise the measurement entry point's provenance and output contracts."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from eo_visual_retrieval.benchmarks import eurosat as benchmark
from eo_visual_retrieval.datasets import eurosat as dataset
from eo_visual_retrieval.hashing import file_md5, file_sha256
from eo_visual_retrieval.manifests import write_jsonl
from eo_visual_retrieval.models import ImageRecord

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eurosat_cell_budget.py"


def _record(item_id: str, *, source: str = dataset.EUROSAT_SOURCE) -> ImageRecord:
    return ImageRecord(
        item_id=item_id,
        path=f"{item_id}.tif",
        split="index",
        label="Forest",
        source=source,
        metadata={"archive_member": "used.tif", "centroid_lonlat": [0.0, 90.0]},
    )


def _invoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, records: list[ImageRecord]
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(records, manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--archive", str(tmp_path / "archive.zip"),
            "--manifest", str(manifest),
            "--output", str(tmp_path / "report.json"),
        ],
    )
    runpy.run_path(str(SCRIPT), run_name="__main__")


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([_record("a"), _record("b")], "duplicate EuroSAT archive members"),
        ([_record("a", source="local")], "record is not a eurosat-ms-v1 item"),
    ],
)
def test_measurement_rejects_invalid_members_before_reading_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    records: list[ImageRecord],
    message: str,
) -> None:
    # No archive exists: provenance errors must be caught before archive access.
    with pytest.raises(ValueError, match=message):
        _invoke(monkeypatch, tmp_path, records)
    assert not (tmp_path / "report.json").exists()


def test_measurement_rejects_unverified_archive_before_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "archive.zip").write_bytes(b"not the official archive")

    def unexpected_discovery(*args: object, **kwargs: object) -> None:
        pytest.fail("discovery ran before checksum verification")

    monkeypatch.setattr(benchmark, "discover_candidates", unexpected_discovery)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _invoke(monkeypatch, tmp_path, [_record("a")])
    assert not (tmp_path / "report.json").exists()


def test_measurement_uses_archive_coordinates_and_records_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"synthetic archive identity")

    # Preserve the real verifier before patching the module used by the script.
    real_verify = dataset.verify_archive

    def verify_fixture(path: Path) -> None:
        real_verify(path, expected_md5=file_md5(archive))

    def discover(path: Path, *, group_size_m: float) -> list[benchmark.EuroSatCandidate]:
        assert path == archive
        assert group_size_m == 50_000
        return [
            benchmark.EuroSatCandidate(
                member=member,
                label="Forest",
                source_crs="EPSG:4326",
                source_bounds=(0.0, 0.0, 1.0, 1.0),
                longitude=0.0,
                latitude=latitude,
                equal_area_x=0.0,
                equal_area_y=latitude * 111_000,
                spatial_group=cell,
            )
            for member, latitude, cell in (
                ("used.tif", 0.0, "cell-1"),
                ("near.tif", 0.1, "cell-1"),
                ("far.tif", 1.0, "cell-2"),
            )
        ]

    monkeypatch.setattr(dataset, "verify_archive", verify_fixture)
    monkeypatch.setattr(benchmark, "discover_candidates", discover)
    _invoke(monkeypatch, tmp_path, [_record("a")])

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["prepared_patches"] == 1
    assert report["unused_patches"] == 2
    assert report["cell_budget"]["free_patches"] == 1
    assert report["distance_from_prepared"]["10km"]["total"] == 2
    assert report["distance_from_prepared"]["20km"]["total"] == 1
    assert report["nearest_km_percentiles"]["p50"] == pytest.approx(61.157)
    assert report["archive_md5"] == file_md5(archive)
    assert report["manifest_sha256"] == file_sha256(tmp_path / "manifest.jsonl")
