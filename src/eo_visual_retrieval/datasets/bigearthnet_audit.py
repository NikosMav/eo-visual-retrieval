"""Offline metadata inventory; no partition selection or spatial-distance claims."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from eo_visual_retrieval.datasets.bigearthnet import (
    BIGEARTHNET_DOI,
    METADATA_ASSETS,
    verify_metadata_file,
)
from eo_visual_retrieval.hashing import file_sha256

_SPLITS = ("train", "validation", "test")
_PATCH_ID = re.compile(
    r"S2[AB]_MSIL2A_(\d{8}T\d{6})_N\d{4}_R\d{3}_T(\d{2}[A-Z]{3})_(\d{2})_(\d{2})"
)
_REQUIRED = {
    "patch_id", "labels", "split", "country", "s1_name", "s2v1_name",
    "contains_seasonal_snow", "contains_cloud_or_shadow",
}


def summarize_rows(rows: Iterable[Mapping[str, Any]], *, excluded: bool) -> dict[str, Any]:
    """Count labels/dates and repeated grid identities, rejecting malformed records.

    Grid identities are tile plus patch row/column. They do not establish metric
    distances or catch footprints shared by adjacent MGRS tiles.
    """
    ids: set[str] = set()
    labels = {split: Counter[str]() for split in _SPLITS}
    months = {split: Counter[str]() for split in _SPLITS}
    countries = {split: Counter[str]() for split in _SPLITS}
    dates: dict[str, set[str]] = {split: set() for split in _SPLITS}
    grid_splits: dict[str, set[str]] = defaultdict(set)
    grid_dates: dict[str, set[str]] = defaultdict(set)
    tiles: set[str] = set()
    flags = Counter[str]()
    for row in rows:
        if not _REQUIRED.issubset(row):
            raise ValueError("metadata row lacks required columns")
        patch_id = row["patch_id"]
        match = _PATCH_ID.fullmatch(patch_id) if isinstance(patch_id, str) else None
        if match is None:
            raise ValueError("invalid BigEarthNet patch ID")
        if patch_id in ids:
            raise ValueError("duplicate metadata patch ID")
        ids.add(patch_id)
        date = datetime.strptime(match[1], "%Y%m%dT%H%M%S").date().isoformat()
        split = row["split"]
        if not isinstance(split, str) or split not in _SPLITS:
            raise ValueError("unknown official split")
        row_labels = row["labels"]
        if not isinstance(row_labels, list) or not row_labels or any(
            not isinstance(label, str) or not label.strip() for label in row_labels
        ):
            raise ValueError("metadata labels must be a non-empty list of strings")
        if len(set(row_labels)) != len(row_labels):
            raise ValueError("duplicate metadata label")
        for field in ("country", "s1_name", "s2v1_name"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"metadata {field} must be a non-empty string")
        snow, cloud = row["contains_seasonal_snow"], row["contains_cloud_or_shadow"]
        if type(snow) is not bool or type(cloud) is not bool:
            raise ValueError("metadata exclusion flags must be booleans")
        if (snow or cloud) != excluded:
            raise ValueError("exclusion flags disagree with the metadata file")
        flags["seasonal_snow"] += snow
        flags["cloud_or_shadow"] += cloud
        flags["both"] += snow and cloud
        labels[split].update(row_labels)
        months[split][date[:7]] += 1
        countries[split][row["country"]] += 1
        dates[split].add(date)
        tile = match[2]
        key = f"{tile}_{match[3]}_{match[4]}"
        tiles.add(tile)
        grid_splits[key].add(split)
        grid_dates[key].add(date)
    if not ids:
        raise ValueError("metadata file is empty")
    all_labels = sorted(set().union(*(set(counts) for counts in labels.values())))
    all_dates = set().union(*dates.values())
    return {
        "rows": len(ids),
        "label_count": len(all_labels),
        "labels": all_labels,
        "date_min": min(all_dates),
        "date_max": max(all_dates),
        "unique_acquisition_dates": len(all_dates),
        "mgrs_tiles": len(tiles),
        "exclusion_flags": dict(sorted(flags.items())),
        "splits": {
            split: {
                "rows": sum(months[split].values()),
                "unique_acquisition_dates": len(dates[split]),
                "date_min": min(dates[split], default=None),
                "date_max": max(dates[split], default=None),
                "months": dict(sorted(months[split].items())),
                "countries": dict(sorted(countries[split].items())),
                "label_counts": {label: labels[split][label] for label in all_labels},
            }
            for split in _SPLITS
        },
        "shared_acquisition_dates": {
            f"{left}/{right}": len(dates[left] & dates[right])
            for left, right in combinations(_SPLITS, 2)
        },
        "grid_identity": {
            "definition": "MGRS tile plus patch row/column; no metric footprint audit",
            "unique_keys": len(grid_splits),
            "keys_in_multiple_splits": sum(len(splits) > 1 for splits in grid_splits.values()),
            "keys_on_multiple_dates": sum(len(days) > 1 for days in grid_dates.values()),
        },
    }


def audit_metadata(directory: Path) -> dict[str, Any]:
    """Verify both local files before parsing; never fetch missing data."""
    for asset in METADATA_ASSETS:
        verify_metadata_file(directory / asset.filename, asset)
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("Install the bigearthnet extra to inspect Parquet metadata") from error
    summaries: dict[str, Any] = {}
    id_sets: list[set[str]] = []
    for asset in METADATA_ASSETS:
        path = directory / asset.filename
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        if not _REQUIRED.issubset(columns):
            raise ValueError("metadata Parquet lacks required columns")
        rows = (
            row for batch in parquet.iter_batches(batch_size=8192)
            for row in batch.to_pylist()
        )
        summary = summarize_rows(rows, excluded=asset.filename != "metadata.parquet")
        summaries[asset.filename] = {
            "bytes": path.stat().st_size,
            "md5": asset.md5,
            "sha256": file_sha256(path),
            "columns": {field.name: str(field.type) for field in parquet.schema_arrow},
            **summary,
        }
        id_sets.append(set(parquet.read(columns=["patch_id"])["patch_id"].to_pylist()))
    if id_sets[0] & id_sets[1]:
        raise ValueError("recommended and excluded metadata overlap")
    return {
        "schema": "bigearthnet-metadata-audit-v1",
        "doi": BIGEARTHNET_DOI,
        "files": summaries,
        "total_patches": sum(len(ids) for ids in id_sets),
        "recommended_excluded_overlap": 0,
        "spatial_footprints_audited": False,
        "partitions_prepared": False,
        "retrieval_scored": False,
    }
