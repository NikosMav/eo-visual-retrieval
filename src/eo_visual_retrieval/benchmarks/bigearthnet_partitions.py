"""Freeze acquisition IDs before imagery or model scores are inspected."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from eo_visual_retrieval.benchmarks.coverage import nearest_distances_m
from eo_visual_retrieval.datasets.bigearthnet import METADATA_ASSETS, REFERENCE_ARCHIVE_FILENAME
from eo_visual_retrieval.datasets.bigearthnet_footprints import (
    CELL_SIZE_M,
    add_centers,
    metadata_identities,
    reference_footprints,
)
from eo_visual_retrieval.hashing import file_sha256

POLICY: dict[str, Any] = {
    "version": "bigearthnet-acquisition-selection-v1",
    "seed": 42,
    "partitions": {
        "index": {"official_split": "train", "start": "2017-06-01", "end": "2017-09-30",
                  "size": 4000},
        "development": {"official_split": "validation", "start": "2017-11-01",
                        "end": "2018-02-28", "size": 500},
        "final": {"official_split": "test", "start": "2018-04-01", "end": "2018-05-31",
                  "size": 500},
    },
    "cell_size_m": CELL_SIZE_M,
    "minimum_query_cells": 20,
    "minimum_label_count": 5,
    "minimum_center_distance_m": 7000,
    "minimum_footprint_separation_lower_bound_m": 5000,
    "minimum_temporal_gap_days": 30,
    "exclude_from_final": ["S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"],
    "algorithm": "final then development then index; rare-label cell allocation; "
                 "seeded cell order and round-robin patch fill; one observation per grid key",
}


def _rank(identity: str, seed: int) -> str:
    # This is a sampling order, not a file/content provenance digest.
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def _day(patch_id: str) -> str:
    value = patch_id.split("_")[2][:8]
    return date(int(value[:4]), int(value[4:6]), int(value[6:8])).isoformat()


def _grid(patch_id: str) -> str:
    return "_".join(patch_id.split("_")[-3:])


def _points(rows: list[dict[str, Any]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.asarray([(row["longitude"], row["latitude"]) for row in rows], dtype=np.float64)


def load_candidates(directory: Path, inventory: Path, report: Path) -> list[dict[str, Any]]:
    """Bind an inventory to verified sources before using its derived geometry."""
    import pyarrow.parquet as pq

    identities = metadata_identities(directory)
    evidence = json.loads(report.read_text(encoding="utf-8"))
    if evidence["inventory_sha256"] != file_sha256(inventory):
        raise ValueError("footprint inventory checksum mismatch")
    if evidence["reference_archive_sha256"] != file_sha256(directory / REFERENCE_ARCHIVE_FILENAME):
        raise ValueError("reference archive differs from inventory source")
    for asset in METADATA_ASSETS:
        if evidence["metadata_sha256"][asset.filename] != file_sha256(directory / asset.filename):
            raise ValueError("metadata differs from inventory source")
    geometry = pq.read_table(inventory).to_pylist()
    by_id = {row["patch_id"]: row for row in geometry}
    if len(by_id) != len(geometry) or set(by_id) != identities:
        raise ValueError("footprint inventory identity coverage mismatch")
    recommended = pq.read_table(directory / "metadata.parquet").to_pylist()
    return [{**row, **by_id[row["patch_id"]]} for row in recommended]


def _sample(
    candidates: list[dict[str, Any]], labels: list[str], size: int, policy: dict[str, Any],
    *, is_index: bool,
) -> list[dict[str, Any]]:
    seed, minimum = policy["seed"], policy["minimum_label_count"]
    ordered = sorted(candidates, key=lambda row: _rank(row["patch_id"], seed))
    # Choose one date per grid before computing capacities, so repeats cannot inflate them.
    unique: dict[str, dict[str, Any]] = {}
    for row in ordered:
        unique.setdefault(_grid(row["patch_id"]), row)
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique.values():
        cells[row["spatial_group"]].append(row)
    capacities = {cell: Counter(label for row in rows for label in row["labels"])
                  for cell, rows in cells.items()}
    totals = Counter(label for row in unique.values() for label in row["labels"])
    rare_first = sorted(labels, key=lambda label: (totals[label], label))
    owned: set[str] = set(cells) if is_index else set()
    coverage: Counter[str] = Counter()
    if not is_index:
        for label in rare_first:
            while coverage[label] < minimum:
                choices = [cell for cell in cells if cell not in owned and capacities[cell][label]]
                if not choices:
                    raise ValueError(f"insufficient spatially eligible label capacity: {label}")
                chosen = min(choices, key=lambda cell: (
                    -min(minimum - coverage[label], capacities[cell][label]), _rank(cell, seed)
                ))
                owned.add(chosen)
                coverage.update(capacities[chosen])
        for cell in sorted(cells, key=lambda cell: _rank(cell, seed)):
            if len(owned) >= policy["minimum_query_cells"] and sum(
                len(cells[value]) for value in owned
            ) >= size:
                break
            owned.add(cell)
        if len(owned) < policy["minimum_query_cells"]:
            raise ValueError("insufficient spatial cells for query partition")
    if sum(len(cells[cell]) for cell in owned) < size:
        raise ValueError("insufficient unique patches for requested partition size")
    if len(owned) > size:
        raise ValueError("partition size cannot cover every allocated spatial cell")
    result: list[dict[str, Any]] = []
    used: set[str] = set()
    counts: Counter[str] = Counter()

    def take(row: dict[str, Any]) -> None:
        result.append(row)
        used.add(row["patch_id"])
        counts.update(row["labels"])

    owned_order = sorted(owned, key=lambda cell: _rank(cell, seed))
    for cell in owned_order:
        take(cells[cell][0])
    pool = [row for row in unique.values() if row["spatial_group"] in owned]
    for label in rare_first:
        for row in pool:
            if counts[label] >= minimum:
                break
            if row["patch_id"] not in used and label in row["labels"]:
                take(row)
        if counts[label] < minimum:
            raise ValueError(f"insufficient selected label coverage: {label}")
    if len(result) > size:
        raise ValueError("partition size cannot satisfy cells and label coverage")
    offset = 0
    while len(result) < size:
        for cell in owned_order:
            if offset < len(cells[cell]):
                row = cells[cell][offset]
                if row["patch_id"] not in used:
                    take(row)
                    if len(result) == size:
                        break
        offset += 1
    return sorted(result, key=lambda row: row["patch_id"])


def select_partitions(
    candidates: list[dict[str, Any]], *, policy: dict[str, Any] = POLICY,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    labels = sorted({label for row in candidates for label in row["labels"]})
    selected: dict[str, list[dict[str, Any]]] = {}
    feasibility: dict[str, Any] = {}
    for partition in ("final", "development", "index"):
        spec = policy["partitions"][partition]
        eligible = [row for row in candidates if row["split"] == spec["official_split"]
                    and spec["start"] <= _day(row["patch_id"]) <= spec["end"]
                    and not (partition == "final"
                             and row["patch_id"] in policy["exclude_from_final"])]
        temporal_count = len(eligible)
        used = [row for rows in selected.values() for row in rows]
        if used:
            used_cells = {row["spatial_group"] for row in used}
            eligible = [row for row in eligible if row["spatial_group"] not in used_cells]
            if eligible:
                distances = nearest_distances_m(_points(eligible), _points(used))
                eligible = [row for row, distance in zip(eligible, distances, strict=True)
                            if distance >= policy["minimum_center_distance_m"]]
        selected[partition] = _sample(
            eligible, labels, spec["size"], policy, is_index=partition == "index",
        )
        feasibility[partition] = {
            "after_temporal_filter": temporal_count,
            "after_spatial_filter": len(eligible),
            "available_cells": len({row["spatial_group"] for row in eligible}),
        }
    return ({part: [row["patch_id"] for row in selected[part]]
             for part in policy["partitions"]}, feasibility)


def audit_selection(
    selection: dict[str, Any], candidates: list[dict[str, Any]],
    *, policy: dict[str, Any] = POLICY,
) -> dict[str, Any]:
    """Independent invariant audit: no use of the selection/ranking algorithm.

    Production callers replace selected geometry with freshly read source TIFF
    headers before calling this audit. Distances use a spherical Earth model.
    """
    from rasterio.warp import transform

    if selection["policy"] != policy or set(selection["partitions"]) != set(policy["partitions"]):
        raise ValueError("selection policy or partition names differ from the frozen protocol")
    by_id = {row["patch_id"]: row for row in candidates}
    labels = sorted({label for row in candidates for label in row["labels"]})
    seen: set[str] = set()
    rows_by_part: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, Any] = {}
    for part, spec in policy["partitions"].items():
        ids = selection["partitions"][part]
        if len(ids) != spec["size"] or len(set(ids)) != len(ids) or seen & set(ids):
            raise ValueError("selection size or identity uniqueness violation")
        if not set(ids) <= by_id.keys():
            raise ValueError("selected ID absent from recommended metadata")
        seen.update(ids)
        rows = [dict(by_id[identity]) for identity in ids]
        add_centers(rows)
        if len({_grid(row["patch_id"]) for row in rows}) != len(rows):
            raise ValueError("repeated grid observation in partition")
        if any(row["contains_seasonal_snow"] or row["contains_cloud_or_shadow"] for row in rows):
            raise ValueError("excluded patch in selection")
        if any(row["split"] != spec["official_split"] or not (
            spec["start"] <= _day(row["patch_id"]) <= spec["end"]
        ) for row in rows):
            raise ValueError("official split or temporal window violation")
        if part == "final" and set(ids) & set(policy["exclude_from_final"]):
            raise ValueError("previously inspected image in final selection")
        counts = Counter(label for row in rows for label in row["labels"])
        if any(counts[label] < policy["minimum_label_count"] for label in labels):
            raise ValueError("insufficient label coverage in selection")
        cells = {row["spatial_group"] for row in rows}
        if part != "index" and len(cells) < policy["minimum_query_cells"]:
            raise ValueError("insufficient query cells in selection")
        radii = []
        for row in rows:
            lon, lat = transform(row["epsg"], 4326,
                                 [row["left"], row["left"], row["right"], row["right"]],
                                 [row["bottom"], row["top"], row["bottom"], row["top"]])
            radii.append(float(nearest_distances_m(
                np.asarray(list(zip(lon, lat, strict=True))), _points([row]),
            ).max()))
        stats[part] = {
            "patches": len(rows), "cells": len(cells),
            "date_min": min(_day(row["patch_id"]) for row in rows),
            "date_max": max(_day(row["patch_id"]) for row in rows),
            "label_counts": {label: counts[label] for label in labels},
            "countries": dict(sorted(Counter(row["country"] for row in rows).items())),
            "maximum_corner_radius_m": max(radii),
        }
        rows_by_part[part] = rows
    pairs = {}
    for left, right in combinations(policy["partitions"], 2):
        if {row["spatial_group"] for row in rows_by_part[left]} & {
            row["spatial_group"] for row in rows_by_part[right]
        }:
            raise ValueError("spatial cell overlap across partitions")
        center = float(nearest_distances_m(_points(rows_by_part[left]),
                                          _points(rows_by_part[right])).min())
        # 1% allowance beyond measured corner radii covers projection/geodesic curvature
        # of these small UTM squares. This is a conservative bound, not polygon distance.
        edge_bound = center - 1.01 * (stats[left]["maximum_corner_radius_m"]
                                      + stats[right]["maximum_corner_radius_m"])
        gap = (date.fromisoformat(stats[right]["date_min"])
               - date.fromisoformat(stats[left]["date_max"])).days
        if center < policy["minimum_center_distance_m"] or edge_bound < (
            policy["minimum_footprint_separation_lower_bound_m"]
        ):
            raise ValueError("spatial guard-band violation")
        if gap < policy["minimum_temporal_gap_days"]:
            raise ValueError("temporal guard-band violation")
        pairs[f"{left}/{right}"] = {
            "shared_cells": 0, "minimum_center_distance_m": center,
            "footprint_separation_lower_bound_m": edge_bound, "temporal_gap_days": gap,
        }
    return {"partitions": stats, "pairwise_separation": pairs,
            "distance_model": "spherical great-circle; maximum corner radii inflated by 1%"}


def audit_selection_from_source(
    selection: dict[str, Any], candidates: list[dict[str, Any]], directory: Path,
    *, policy: dict[str, Any] = POLICY,
) -> dict[str, Any]:
    ids = {identity for values in selection["partitions"].values() for identity in values}
    fresh = {row["patch_id"]: row for row in reference_footprints(
        directory / REFERENCE_ARCHIVE_FILENAME, metadata_identities(directory), selected_ids=ids,
    )}
    by_id = {row["patch_id"]: row for row in candidates}
    geometry_fields = ("epsg", "left", "bottom", "right", "top")
    for identity, geometry in fresh.items():
        if identity not in by_id or any(
            geometry[key] != by_id[identity][key] for key in geometry_fields
        ):
            raise ValueError("source geometry differs from footprint inventory")
    rows = [{**row, **fresh.get(row["patch_id"], {})} for row in candidates]
    result = audit_selection(selection, rows, policy=policy)
    return {**result, "selected_reference_footprints_reloaded": len(fresh),
            "independent_geometry_readers_agree": True,
            "s2_subset_footprints_verified": False, "retrieval_scored": False}
