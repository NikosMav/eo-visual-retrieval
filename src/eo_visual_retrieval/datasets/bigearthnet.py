"""Pinned BigEarthNet v2 identities and bounded acquisition of metadata only."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from eo_visual_retrieval.hashing import file_md5, file_sha256

BIGEARTHNET_DOI = "10.5281/zenodo.10891137"
BIGEARTHNET_RECORD_URL = "https://zenodo.org/records/10891137"
BIGEARTHNET_LICENSE = "CDLA-Permissive-1.0"
S2_ARCHIVE_FILENAME = "BigEarthNet-S2.tar.zst"
S2_ARCHIVE_MD5 = "2245ed2d1a93f6ce637d839bc856396e"
S2_ARCHIVE_BYTES = 63_251_710_377
# Explicit native L2A order. B10 is absent; this is not EuroSAT's 13-band layout.
S2_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
S2_GSD = {band: (60 if band in ("B01", "B09") else 10 if band in (
    "B02", "B03", "B04", "B08"
) else 20) for band in S2_BANDS}
REFERENCE_ARCHIVE_FILENAME = "Reference_Maps.tar.zst"
REFERENCE_ARCHIVE_MD5 = "95d85a222fa983faddcac51a19f28917"
REFERENCE_ARCHIVE_BYTES = 282_391_301


@dataclass(frozen=True)
class MetadataAsset:
    filename: str
    md5: str
    advertised_size: str


METADATA_ASSETS = (
    MetadataAsset("metadata.parquet", "55687065e77b6d0b0f1ff604a6e7b49c", "3.6 MB"),
    MetadataAsset(
        "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
        "fe31856f4986d446c9468b59d6387c91",
        "710.2 kB",
    ),
)
# This reader cannot fetch imagery. Two metadata files are each capped at 8 MiB.
MAX_METADATA_BYTES = 8 * 1024 * 1024


def acquisition_plan() -> dict[str, Any]:
    """Published inventory, not a claim that any local file has been verified."""
    return {
        "dataset": "BigEarthNet-v2.0.0",
        "doi": BIGEARTHNET_DOI,
        "record": BIGEARTHNET_RECORD_URL,
        "license": BIGEARTHNET_LICENSE,
        "s2_archive": {
            "filename": S2_ARCHIVE_FILENAME,
            "published_md5": S2_ARCHIVE_MD5,
            "published_bytes": S2_ARCHIVE_BYTES,
            "advertised_size": "63.3 GB (approximately 59 GiB)",
            "download_enabled": False,
        },
        "metadata": [
            {
                "filename": asset.filename,
                "published_md5": asset.md5,
                "advertised_size": asset.advertised_size,
            }
            for asset in METADATA_ASSETS
        ],
        "max_metadata_bytes_per_file": MAX_METADATA_BYTES,
        "cost_ceiling": "local CPU and disk; no paid service",
    }


def verify_metadata_file(path: Path, asset: MetadataAsset) -> None:
    """Verify a local metadata file against its pinned identity and byte ceiling."""
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError(f"metadata exceeds byte limit: {asset.filename}")
    if file_md5(path) != asset.md5:
        raise ValueError(f"metadata checksum mismatch: {asset.filename}")


def download_metadata(output_dir: Path) -> list[dict[str, Any]]:
    """Download only allowlisted metadata, verifying before atomic promotion.

Valid cached files are reused. A corrupt existing destination is refused; a
failed or oversized response is never promoted into the local dataset.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for asset in METADATA_ASSETS:
        destination = output_dir / asset.filename
        if destination.exists():
            verify_metadata_file(destination, asset)
        else:
            url = f"{BIGEARTHNET_RECORD_URL}/files/{asset.filename}?download=1"
            with urlopen(url, timeout=30) as response:
                payload = response.read(MAX_METADATA_BYTES + 1)
            if len(payload) > MAX_METADATA_BYTES:
                raise ValueError(f"metadata exceeds byte limit: {asset.filename}")
            with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                temporary.write_bytes(payload)
                verify_metadata_file(temporary, asset)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        results.append({
            "filename": asset.filename,
            "bytes": destination.stat().st_size,
            "md5": file_md5(destination),
            "sha256": file_sha256(destination),
        })
    return results
