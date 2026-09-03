"""Metadata dates and grid identities must not overstate partition independence."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from eo_visual_retrieval.datasets import bigearthnet_audit as audit
from eo_visual_retrieval.datasets.bigearthnet import MetadataAsset
from eo_visual_retrieval.hashing import file_md5


def _row(
    day: str = "20170613", split: str = "train", cell: str = "26_57", **kwargs: Any
) -> dict[str, Any]:
    return {
        "patch_id": f"S2A_MSIL2A_{day}T101031_N9999_R022_T33UUP_{cell}",
        "labels": ["Arable land", "Pastures"],
        "split": split,
        "country": "Austria",
        "s1_name": "paired-s1",
        "s2v1_name": "original-s2",
        "contains_seasonal_snow": False,
        "contains_cloud_or_shadow": False,
        **kwargs,
    }


def test_dates_and_grid_repeats_are_measured_without_claiming_spatial_audit() -> None:
    rows = [
        _row(),
        _row("20170701"),
        _row("20170801", "validation"),
        _row("20170613", "test", "27_57", labels=["Pastures"]),
    ]
    result = audit.summarize_rows(iter(rows), excluded=False)
    assert result["rows"] == 4
    assert result["label_count"] == 2
    assert result["date_min"] == "2017-06-13"
    assert result["date_max"] == "2017-08-01"
    assert result["splits"]["train"]["months"] == {"2017-06": 1, "2017-07": 1}
    assert result["splits"]["train"]["label_counts"]["Pastures"] == 2
    assert result["splits"]["test"]["label_counts"]["Arable land"] == 0
    assert result["shared_acquisition_dates"]["train/test"] == 1
    assert result["shared_acquisition_dates"]["train/validation"] == 0
    assert result["grid_identity"]["unique_keys"] == 2
    assert result["grid_identity"]["keys_in_multiple_splits"] == 1
    assert result["grid_identity"]["keys_on_multiple_dates"] == 1
    assert "no metric footprint audit" in result["grid_identity"]["definition"]
    assert audit.summarize_rows(reversed(rows), excluded=False) == result


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"patch_id": "bad"}, "patch ID"),
        ({"patch_id": None}, "patch ID"),
        ({"split": "development"}, "official split"),
        ({"labels": []}, "labels"),
        ({"labels": "Arable land"}, "labels"),
        ({"labels": [None]}, "labels"),
        ({"labels": ["Pastures", "Pastures"]}, "duplicate metadata label"),
        ({"country": None}, "country"),
        ({"contains_seasonal_snow": "false"}, "booleans"),
        ({"contains_cloud_or_shadow": True}, "exclusion flags disagree"),
    ],
)
def test_invalid_metadata_is_rejected(changes: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        audit.summarize_rows([_row(**changes)], excluded=False)


def test_empty_duplicate_missing_and_invalid_calendar_rows_are_rejected() -> None:
    for rows, error in [
        ([], "empty"),
        ([_row(), _row()], "duplicate metadata patch ID"),
        ([{"patch_id": "bad"}], "required columns"),
        ([_row("20170230")], "day is out of range"),
    ]:
        with pytest.raises(ValueError, match=error):
            audit.summarize_rows(rows, excluded=False)


def test_exclusion_counts_are_explicit_and_absent_splits_have_no_dates() -> None:
    result = audit.summarize_rows(
        [_row(contains_seasonal_snow=True, contains_cloud_or_shadow=True)], excluded=True
    )
    assert result["exclusion_flags"] == {"both": 1, "cloud_or_shadow": 1, "seasonal_snow": 1}
    assert result["splits"]["test"]["rows"] == 0
    assert result["splits"]["test"]["date_min"] is None


def _parquet_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, overlap: bool = False
) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    directory = tmp_path / "metadata"
    directory.mkdir()
    assets = []
    for original in audit.METADATA_ASSETS:
        excluded = original.filename != "metadata.parquet"
        row = _row(
            cell="26_57" if not excluded or overlap else "27_57",
            contains_seasonal_snow=excluded,
        )
        path = directory / original.filename
        pq.write_table(pa.Table.from_pylist([row]), path)
        assets.append(MetadataAsset(path.name, file_md5(path), "synthetic"))
    monkeypatch.setattr(audit, "METADATA_ASSETS", tuple(assets))
    return directory


def test_parquet_audit_binds_inputs_and_keeps_excluded_records_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _parquet_files(tmp_path, monkeypatch)
    result = audit.audit_metadata(directory)
    assert result["total_patches"] == 2
    assert result["recommended_excluded_overlap"] == 0
    assert result["spatial_footprints_audited"] is False
    assert result["partitions_prepared"] is False
    assert result["retrieval_scored"] is False
    for summary in result["files"].values():
        assert len(summary["sha256"]) == 64
        assert summary["rows"] == 1
    # Checksum rejection precedes any attempt to parse the replaced bytes.
    (directory / "metadata.parquet").write_bytes(b"not parquet")
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit.audit_metadata(directory)


def test_recommended_excluded_overlap_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _parquet_files(tmp_path, monkeypatch, overlap=True)
    with pytest.raises(ValueError, match="metadata overlap"):
        audit.audit_metadata(directory)


def test_missing_metadata_fails_without_attempting_acquisition(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        audit.audit_metadata(tmp_path)


def test_audit_script_writes_report_and_refuses_output_inside_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _parquet_files(tmp_path, monkeypatch)
    output = tmp_path / "audit.json"
    script = Path(__file__).resolve().parents[1] / "scripts/audit_bigearthnet_metadata.py"
    argv = [str(script), "--metadata-dir", str(directory), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(script), run_name="__main__")
    assert json.loads(output.read_text()) == json.loads(capsys.readouterr().out)
    monkeypatch.setattr(sys, "argv", [*argv[:-1], str(directory / "metadata.parquet")])
    with pytest.raises(SystemExit):
        runpy.run_path(str(script), run_name="__main__")
