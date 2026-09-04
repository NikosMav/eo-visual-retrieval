"""How much of a dataset's geography a prepared benchmark has consumed.

Split preparation, split auditing, and coverage reporting all need the same
great-circle distance. Keeping one implementation here means a guard band that
passes an audit cannot be measured differently from the guard band that was
enforced during preparation.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, types only
    from eo_visual_retrieval.benchmarks.eurosat import EuroSatCandidate

EARTH_RADIUS_M = 6_371_008.8

# Rows per block when comparing two point sets. The pairwise matrix is
# len(block) x len(right) floats, so this bounds peak memory for large inputs
# without giving up vectorisation.
_BLOCK_ROWS = 512


def _as_lonlat(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2) of longitude and latitude")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} coordinates must be finite")
    if np.any(np.abs(array[:, 0]) > 180) or np.any(np.abs(array[:, 1]) > 90):
        raise ValueError(f"{name} coordinates exceed longitude/latitude bounds")
    return array


def nearest_distances_m(
    left_lonlat: NDArray[np.float64], right_lonlat: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Great-circle metres from each left point to its nearest right point."""

    left = np.radians(_as_lonlat(left_lonlat, "left_lonlat"))
    right = np.radians(_as_lonlat(right_lonlat, "right_lonlat"))
    out = np.empty(len(left), dtype=np.float64)
    for start in range(0, len(left), _BLOCK_ROWS):
        block = left[start : start + _BLOCK_ROWS]
        delta_lon = right[None, :, 0] - block[:, None, 0]
        delta_lat = right[None, :, 1] - block[:, None, 1]
        haversine = np.sin(delta_lat / 2) ** 2 + (
            np.cos(block[:, None, 1]) * np.cos(right[None, :, 1]) * np.sin(delta_lon / 2) ** 2
        )
        distances = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
        out[start : start + _BLOCK_ROWS] = distances.min(axis=1)
    return out


@dataclass(frozen=True)
class CellBudget:
    """How much of a dataset's cell geography a prepared benchmark consumed."""

    total_patches: int
    total_cells: int
    used_patches: int
    used_cells: int
    free_cells: int
    free_patches: int
    per_label: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_patches": self.total_patches,
            "total_cells": self.total_cells,
            "used_patches": self.used_patches,
            "used_cells": self.used_cells,
            "free_cells": self.free_cells,
            "free_patches": self.free_patches,
            "per_label": self.per_label,
        }


def cell_budget(
    candidates: Sequence[EuroSatCandidate], used_members: set[str]
) -> CellBudget:
    """Measure the cells left untouched after a benchmark selected ``used_members``.

    A cell is spent as soon as one of its patches is selected: a later partition
    drawn from the same cell would not be spatially separated from the earlier
    one, which is the whole point of grouping by cell.
    """

    if not candidates:
        raise ValueError("at least one candidate is required")
    members = {candidate.member for candidate in candidates}
    unknown = sorted(used_members - members)
    if unknown:
        raise ValueError(f"used member is not present among the candidates: {unknown[0]}")

    used_cells = {
        candidate.spatial_group for candidate in candidates if candidate.member in used_members
    }
    all_cells = {candidate.spatial_group for candidate in candidates}
    free = [candidate for candidate in candidates if candidate.spatial_group not in used_cells]

    free_patches_by_label: collections.Counter[str] = collections.Counter()
    free_cells_by_label: dict[str, set[str]] = collections.defaultdict(set)
    for candidate in free:
        free_patches_by_label[candidate.label] += 1
        free_cells_by_label[candidate.label].add(candidate.spatial_group)

    return CellBudget(
        total_patches=len(candidates),
        total_cells=len(all_cells),
        used_patches=len(used_members),
        used_cells=len(used_cells),
        free_cells=len(all_cells - used_cells),
        free_patches=len(free),
        per_label={
            label: {
                "free_patches": free_patches_by_label[label],
                "free_cells": len(free_cells_by_label[label]),
            }
            for label in sorted({candidate.label for candidate in candidates})
        },
    )


def _unused_nearest_m(
    candidates: Sequence[EuroSatCandidate],
    used_members: set[str],
    reference_lonlat: NDArray[np.float64],
) -> tuple[list[EuroSatCandidate], NDArray[np.float64]]:
    """Return unused candidates and each one's nearest distance to ``reference_lonlat``.

    Shared by :func:`distance_tiers` and :func:`nearest_distance_percentiles` so the
    unused-member filter, the coordinate array, and the nearest-distance pass are
    computed once rather than twice.
    """

    members = {candidate.member for candidate in candidates}
    unknown = sorted(used_members - members)
    if unknown:
        raise ValueError(f"used member is not present among the candidates: {unknown[0]}")

    unused = [candidate for candidate in candidates if candidate.member not in used_members]
    if not unused:
        raise ValueError("every candidate was used; no unused patch remains to measure")
    unused_lonlat = np.asarray(
        [(candidate.longitude, candidate.latitude) for candidate in unused], dtype=np.float64
    )
    return unused, nearest_distances_m(unused_lonlat, reference_lonlat)


def distance_tiers(
    candidates: Sequence[EuroSatCandidate],
    used_members: set[str],
    reference_lonlat: NDArray[np.float64],
    thresholds_km: Sequence[float],
) -> dict[str, Any]:
    """Count unused patches surviving a minimum distance from a reference set.

    This is the weaker fallback when cell disjointness is unavailable: it keeps
    patches far from anything already used, without guaranteeing they occupy
    cells the earlier benchmark never touched.
    """

    if not candidates:
        raise ValueError("at least one candidate is required")
    if not thresholds_km or any(not np.isfinite(value) or value <= 0 for value in thresholds_km):
        raise ValueError("thresholds_km must be positive and finite")

    unused, nearest = _unused_nearest_m(candidates, used_members, reference_lonlat)

    tiers: dict[str, Any] = {}
    for threshold in thresholds_km:
        keep = nearest >= threshold * 1000
        patches: collections.Counter[str] = collections.Counter()
        cells: dict[str, set[str]] = collections.defaultdict(set)
        for candidate, is_kept in zip(unused, keep, strict=True):
            if is_kept:
                patches[candidate.label] += 1
                cells[candidate.label].add(candidate.spatial_group)
        per_label = {
            label: {"patches": patches[label], "cells": len(cells[label])}
            for label in sorted({candidate.label for candidate in candidates})
        }
        tiers[f"{threshold:g}km"] = {
            "total": int(keep.sum()),
            "labels_present": sum(1 for value in per_label.values() if value["patches"] > 0),
            "min_label_patches": min(value["patches"] for value in per_label.values()),
            "per_label": per_label,
        }
    return tiers


def nearest_distance_percentiles(
    candidates: Sequence[EuroSatCandidate],
    used_members: set[str],
    reference_lonlat: NDArray[np.float64],
    percentiles: Sequence[float] = (5, 25, 50, 75, 95),
) -> dict[str, float]:
    """Summarise how far unused patches sit from an already-used reference set.

    Distance tiers answer "how many survive a cutoff"; this answers "how far
    is the typical one", which is what makes a weak fallback's weakness
    legible: a low median means most unused patches sit right next to
    something already used, however many clear an arbitrary threshold.
    """

    if not candidates:
        raise ValueError("at least one candidate is required")
    if not percentiles:
        raise ValueError("percentiles must not be empty")

    _unused, nearest = _unused_nearest_m(candidates, used_members, reference_lonlat)
    nearest_km = nearest / 1000

    return {
        f"p{value:g}": round(float(np.percentile(nearest_km, value)), 3) for value in percentiles
    }
