"""How much of a dataset's geography a prepared benchmark has consumed.

Split preparation, split auditing, and coverage reporting all need the same
great-circle distance. Keeping one implementation here means a guard band that
passes an audit cannot be measured differently from the guard band that was
enforced during preparation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

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
