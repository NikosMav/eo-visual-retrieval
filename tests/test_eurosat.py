from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from eo_visual_retrieval.benchmarks.eurosat import (
    EuroSatCandidate,
    audit_eurosat_manifest,
    prepare_eurosat_benchmark,
    select_spatial_split,
)
from eo_visual_retrieval.manifests import read_jsonl


def _candidate(label: str, index: int, *, y_offset: float) -> EuroSatCandidate:
    x = index * 100_000.0
    y = y_offset
    return EuroSatCandidate(
        member=f"EuroSAT_MS/{label}/{label}_{index}.tif",
        label=label,
        source_crs="EPSG:6933",
        source_bounds=(x, y, x + 640, y + 640),
        longitude=x / 111_320,
        latitude=y / 111_320,
        equal_area_x=x,
        equal_area_y=y,
        spatial_group=f"epsg6933:{index}:{round(y_offset)}",
    )


def test_spatial_split_is_balanced_deterministic_and_separated() -> None:
    candidates = [
        *[_candidate("a", index, y_offset=0) for index in range(8)],
        *[_candidate("b", index, y_offset=1_000_000) for index in range(8)],
    ]

    first = select_spatial_split(
        candidates,
        queries_per_class=2,
        index_per_class=3,
        minimum_separation_m=10_000,
        seed=7,
        labels=("a", "b"),
    )
    second = select_spatial_split(
        candidates,
        queries_per_class=2,
        index_per_class=3,
        minimum_separation_m=10_000,
        seed=7,
        labels=("a", "b"),
    )

    assert first == second
    assert {label: sum(item.label == label for item in first.query) for label in ("a", "b")} == {
        "a": 2,
        "b": 2,
    }
    assert {label: sum(item.label == label for item in first.index) for label in ("a", "b")} == {
        "a": 3,
        "b": 3,
    }
    assert {item.spatial_group for item in first.query}.isdisjoint(
        item.spatial_group for item in first.index
    )
    assert first.minimum_separation_m >= 10_000


def _geotiff_bytes(x: float, y: float) -> bytes:
    pytest.importorskip("rasterio")
    from rasterio.enums import ColorInterp
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    data = np.zeros((13, 8, 8), dtype=np.uint16)
    data[3] = 2750
    data[2] = 1375
    data[1] = 0
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            width=8,
            height=8,
            count=13,
            dtype="uint16",
            crs="EPSG:32631",
            transform=from_origin(x, y, 10, 10),
        ) as dataset:
            dataset.write(data)
            dataset.colorinterp = (ColorInterp.undefined,) * 13
        return memory_file.read()


def _make_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for label_index, label in enumerate(("AnnualCrop", "Forest")):
            for index in range(8):
                x = 150_000 + index * 50_000
                y = 5_500_000 + label_index * 500_000
                bundle.writestr(
                    f"EuroSAT_MS/{label}/{label}_{index}.tif",
                    _geotiff_bytes(x, y),
                )


def test_prepare_eurosat_writes_rgb_images_and_auditable_manifest(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    archive = tmp_path / "EuroSAT_MS.zip"
    _make_archive(archive)
    images = tmp_path / "images"
    manifest = tmp_path / "manifest.jsonl"

    result = prepare_eurosat_benchmark(
        archive,
        output_dir=images,
        manifest=manifest,
        queries_per_class=1,
        index_per_class=2,
        group_size_m=20_000,
        minimum_separation_m=1_000,
        seed=9,
        expected_md5=None,
        labels=("AnnualCrop", "Forest"),
    )

    records = read_jsonl(manifest)
    assert len(records) == 6
    assert result.discovered == 16
    assert sum(record.split == "query" for record in records) == 2
    assert sum(record.split == "index" for record in records) == 4
    assert {record.label for record in records} == {"AnnualCrop", "Forest"}
    assert {record.source for record in records} == {"eurosat-ms-v1"}
    assert {record.metadata["dataset_doi"] for record in records} == {
        "10.5281/zenodo.7711810"
    }
    query_groups = {
        record.metadata["spatial_group"] for record in records if record.split == "query"
    }
    index_groups = {
        record.metadata["spatial_group"] for record in records if record.split == "index"
    }
    assert query_groups.isdisjoint(index_groups)

    audit = audit_eurosat_manifest(
        manifest,
        image_root=images,
        expected_labels=("AnnualCrop", "Forest"),
        expected_index_per_class=2,
        expected_queries_per_class=1,
    )
    assert audit.items == 6
    assert audit.index == 4
    assert audit.query == 2
    assert audit.verified_files == 6
    assert audit.minimum_separation_m >= 1_000

    sample_path = images / records[0].path
    with rasterio.open(sample_path) as sample:
        assert sample.count == 3
        assert sample.dtypes == ("uint8", "uint8", "uint8")
        assert sample.crs.to_string() == "EPSG:32631"
        assert sample.colorinterp[:3] == (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
        )
        assert sample.read()[:, 0, 0].tolist() == [255, 128, 0]


def test_prepare_eurosat_rejects_wrong_archive_checksum(tmp_path: Path) -> None:
    archive = tmp_path / "EuroSAT_MS.zip"
    archive.write_bytes(b"not-the-official-archive")

    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_eurosat_benchmark(
            archive,
            output_dir=tmp_path / "images",
            manifest=tmp_path / "manifest.jsonl",
            expected_md5="0" * 32,
        )
