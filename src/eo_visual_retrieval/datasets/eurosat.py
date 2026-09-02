"""Identity and raw access for the official georeferenced EuroSAT archive.

This module is a leaf: it depends on nothing else in the package. Both the
benchmark builder and the multispectral encoders need the same archive checksum,
the same source tag, and the same band ordering, so those facts live here rather
than inside whichever component happened to need them first. That keeps the
encoders in ``embeddings/`` from importing the benchmark package.
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from eo_visual_retrieval.hashing import file_md5
from eo_visual_retrieval.models import ImageRecord

EUROSAT_DOI = "10.5281/zenodo.7711810"
EUROSAT_ARCHIVE = "EuroSAT_MS.zip"
EUROSAT_ARCHIVE_MD5 = "091174add3c8e680a49244acf185b9f0"
EUROSAT_SOURCE = "eurosat-ms-v1"
EUROSAT_CLASSES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)
EUROSAT_CLASSES_SET = frozenset(EUROSAT_CLASSES)

# Band order as the archive stores it. The archive puts B8A last; Sentinel-2
# models generally expect it between B08 and B09, so encoders reorder with
# ``band_indices`` rather than assuming a layout.
EUROSAT_BAND_ORDER = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B10",
    "B11",
    "B12",
    "B8A",
)


def band_indices(target_order: Sequence[str]) -> tuple[int, ...]:
    """Map a model's expected band order onto archive band positions."""

    unknown = [band for band in target_order if band not in EUROSAT_BAND_ORDER]
    if unknown:
        raise ValueError(f"unknown EuroSAT band: {unknown[0]}")
    if len(set(target_order)) != len(target_order):
        raise ValueError("target band order must not repeat a band")
    return tuple(EUROSAT_BAND_ORDER.index(band) for band in target_order)


def verify_archive(archive: Path, expected_md5: str | None = EUROSAT_ARCHIVE_MD5) -> None:
    """Reject a missing or unexpected archive before any member is read."""

    if not archive.is_file():
        raise ValueError(f"EuroSAT archive does not exist: {archive}")
    if expected_md5 is None:
        return
    observed = file_md5(archive)
    if observed.lower() != expected_md5.lower():
        raise ValueError(
            f"EuroSAT archive checksum mismatch: expected {expected_md5}, observed {observed}"
        )


def archive_members(records: Sequence[ImageRecord]) -> list[str]:
    """Return the archive member of every record, refusing foreign or duplicate rows."""

    if not records:
        raise ValueError("at least one image record is required")
    members: list[str] = []
    for record in records:
        if record.source != EUROSAT_SOURCE:
            raise ValueError(f"record is not a {EUROSAT_SOURCE} item: {record.item_id}")
        member = record.metadata.get("archive_member")
        if not isinstance(member, str) or not member:
            raise ValueError(f"record lacks an archive member: {record.item_id}")
        members.append(member)
    if len(set(members)) != len(members):
        raise ValueError("manifest contains duplicate EuroSAT archive members")
    return members


def read_archive_member(bundle: zipfile.ZipFile, member: str) -> NDArray[np.generic]:
    """Read one 13-band member as raw digital numbers without touching disk."""

    try:
        from rasterio.io import MemoryFile
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("EuroSAT archive access requires the 'geo' dependency group") from error

    try:
        payload = bundle.read(member)
    except KeyError as error:
        raise ValueError(f"EuroSAT archive member does not exist: {member}") from error
    with MemoryFile(payload) as memory_file, memory_file.open() as dataset:
        if dataset.count != len(EUROSAT_BAND_ORDER):
            raise ValueError(f"EuroSAT archive member does not have 13 bands: {member}")
        return dataset.read()
