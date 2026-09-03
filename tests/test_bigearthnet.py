"""Acquisition remains bounded and cannot promote unverified dataset metadata."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from eo_visual_retrieval.datasets import bigearthnet
from eo_visual_retrieval.hashing import file_md5


def _asset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> bigearthnet.MetadataAsset:
    fixture = tmp_path / "source.parquet"
    fixture.write_bytes(b"synthetic parquet fixture")
    asset = bigearthnet.MetadataAsset("metadata.parquet", file_md5(fixture), "fixture")
    monkeypatch.setattr(bigearthnet, "METADATA_ASSETS", (asset,))
    return asset


def test_plan_records_source_identity_without_enabling_imagery_download() -> None:
    plan = bigearthnet.acquisition_plan()
    assert plan["doi"] == "10.5281/zenodo.10891137"
    assert plan["license"] == "CDLA-Permissive-1.0"
    assert plan["s2_archive"]["download_enabled"] is False
    assert len(plan["metadata"]) == 2
    assert plan["max_metadata_bytes_per_file"] == 8 * 1024 * 1024


def test_metadata_download_verifies_and_reuses_cached_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asset = _asset(monkeypatch, tmp_path)
    calls: list[str] = []

    def open_fixture(url: str, *, timeout: int) -> io.BytesIO:
        assert timeout == 30
        assert url.endswith("/metadata.parquet?download=1")
        calls.append(url)
        return io.BytesIO(b"synthetic parquet fixture")

    monkeypatch.setattr(bigearthnet, "urlopen", open_fixture)
    output_dir = tmp_path / "downloaded"
    first = bigearthnet.download_metadata(output_dir)
    assert first[0]["md5"] == asset.md5
    assert first[0]["bytes"] == 25
    assert len(first[0]["sha256"]) == 64
    assert bigearthnet.download_metadata(output_dir) == first
    assert len(calls) == 1
    assert sorted(path.name for path in output_dir.iterdir()) == ["metadata.parquet"]


@pytest.mark.parametrize("payload", [b"wrong checksum", b"oversized" * 100])
def test_bad_responses_do_not_create_a_dataset_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: bytes
) -> None:
    _asset(monkeypatch, tmp_path)
    monkeypatch.setattr(bigearthnet, "MAX_METADATA_BYTES", 100)
    monkeypatch.setattr(bigearthnet, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    output_dir = tmp_path / "downloaded"
    with pytest.raises(ValueError, match="checksum mismatch|byte limit"):
        bigearthnet.download_metadata(output_dir)
    assert list(output_dir.iterdir()) == []


def test_existing_bad_metadata_is_preserved_and_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _asset(monkeypatch, tmp_path)
    output_dir = tmp_path / "downloaded"
    output_dir.mkdir()
    destination = output_dir / "metadata.parquet"
    destination.write_bytes(b"corrupt existing file")
    with pytest.raises(ValueError, match="checksum mismatch"):
        bigearthnet.download_metadata(output_dir)
    assert destination.read_bytes() == b"corrupt existing file"
