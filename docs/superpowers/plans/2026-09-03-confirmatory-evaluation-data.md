# Confirmatory Evaluation Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish ADR 0006 naming BigEarthNet v2 as the confirmatory evaluation set, backed by a committed, re-runnable measurement proving EuroSAT can no longer supply one.

**Architecture:** A new leaf module `benchmarks/coverage.py` owns one vectorised great-circle implementation plus two measurements — cell budget and distance tiers. `benchmarks/eurosat.py` is refactored to consume that implementation instead of its own private copy, so the project keeps one haversine. `scripts/eurosat_cell_budget.py` is a thin CLI over the module, following the shape of `scripts/validate_gpu.py`. The remaining tasks are documentation: the ADR, the validation record, and the roadmap.

**Tech Stack:** Python 3.11/3.12, NumPy, Rasterio (via the existing `discover_candidates`), pytest, Ruff, Mypy.

## Global Constraints

- Line length 100; Ruff `select = ["E", "F", "I", "UP", "B"]` must pass.
- Mypy runs over `src`, `tests`, `scripts` with `disallow_untyped_defs`; every function needs annotations.
- Coverage must stay at or above 75%.
- No published EuroSAT v1 result may change. `docs/results/*.json` are not edited by this plan.
- `docs/validation.md` records only executed evidence; never write a number that was not produced by a run.
- Never commit anything under `data/`, `artifacts/`, or `outputs/` — all are gitignored.
- Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- The local full environment is `C:\Users\nikos\.venvs\eovr\Scripts\python.exe`. It has rasterio and the EuroSAT archive available at `data/downloads/EuroSAT_MS.zip`.
- Pre-registered values from the spec, to be copied verbatim into the ADR: relevance threshold **tau = 0.5**; sensitivity reported at **0.3, 0.5, 0.7** on the development partition only; partitions **4,000 index / 500 development / 500 final = 5,000 patches**.

## File Structure

| File | Responsibility |
|---|---|
| `src/eo_visual_retrieval/benchmarks/coverage.py` | **Create.** Vectorised great-circle distances; `cell_budget()` and `distance_tiers()` measurements. |
| `src/eo_visual_retrieval/benchmarks/eurosat.py` | **Modify.** Delete the private haversine, delegate to `coverage`. |
| `tests/test_coverage.py` | **Create.** Unit tests for all three public functions. |
| `scripts/eurosat_cell_budget.py` | **Create.** CLI that runs the measurement against the real archive and writes JSON. |
| `docs/decisions/0006-confirmatory-evaluation-data.md` | **Create.** The ADR. |
| `docs/validation.md` | **Modify.** New executed-evidence section. |
| `docs/project-context.md` | **Modify.** Roadmap and next-task update. |
| `README.md` | **Modify.** Roadmap paragraph, which duplicates project-context. |

---

### Task 1: One great-circle implementation

Removes the private haversine from `eurosat.py` and moves a vectorised version into a new module. This must not change any split or audit result.

**Files:**
- Create: `src/eo_visual_retrieval/benchmarks/coverage.py`
- Modify: `src/eo_visual_retrieval/benchmarks/eurosat.py` (imports; delete `_distances_m` at lines 219-229; rewrite `_minimum_distance` at lines 232-241; update call site at line 330)
- Test: `tests/test_coverage.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `EARTH_RADIUS_M: float`
  - `nearest_distances_m(left_lonlat: NDArray[np.float64], right_lonlat: NDArray[np.float64]) -> NDArray[np.float64]` — for each row of `left_lonlat` (columns are longitude, latitude in degrees), the great-circle metres to the nearest row of `right_lonlat`. Returns shape `(len(left_lonlat),)`. Raises `ValueError` on a non-`(n, 2)` input or an empty `right_lonlat`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_coverage.py`:

```python
"""Geographic coverage measurements shared by benchmark preparation and audits."""

from __future__ import annotations

import numpy as np
import pytest

from eo_visual_retrieval.benchmarks.coverage import EARTH_RADIUS_M, nearest_distances_m


def test_nearest_distance_is_zero_for_a_coincident_point() -> None:
    left = np.asarray([[10.0, 45.0]])
    right = np.asarray([[10.0, 45.0], [11.0, 46.0]])

    assert nearest_distances_m(left, right)[0] == pytest.approx(0.0, abs=1e-6)


def test_nearest_distance_matches_a_known_meridian_arc() -> None:
    """One degree of latitude is the earth radius times one degree in radians."""
    left = np.asarray([[0.0, 0.0]])
    right = np.asarray([[0.0, 1.0]])
    expected = EARTH_RADIUS_M * np.radians(1.0)

    assert nearest_distances_m(left, right)[0] == pytest.approx(expected, rel=1e-9)


def test_nearest_distance_picks_the_closest_of_many() -> None:
    left = np.asarray([[0.0, 0.0]])
    right = np.asarray([[0.0, 5.0], [0.0, 1.0], [0.0, 9.0]])
    expected = EARTH_RADIUS_M * np.radians(1.0)

    assert nearest_distances_m(left, right)[0] == pytest.approx(expected, rel=1e-9)


def test_nearest_distances_returns_one_value_per_left_row() -> None:
    left = np.asarray([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    right = np.asarray([[0.0, 0.0]])

    actual = nearest_distances_m(left, right)

    assert actual.shape == (3,)
    np.testing.assert_allclose(
        actual, EARTH_RADIUS_M * np.radians([0.0, 2.0, 4.0]), rtol=1e-9
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.zeros((1, 3)), np.zeros((1, 2))),
        (np.zeros((1, 2)), np.zeros((1, 3))),
        (np.zeros(2), np.zeros((1, 2))),
        (np.zeros((1, 2)), np.zeros((0, 2))),
    ],
)
def test_nearest_distances_rejects_malformed_input(
    left: np.ndarray, right: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        nearest_distances_m(left, right)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest tests/test_coverage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eo_visual_retrieval.benchmarks.coverage'`

- [ ] **Step 3: Write the module**

Create `src/eo_visual_retrieval/benchmarks/coverage.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest tests/test_coverage.py -q`
Expected: PASS, 8 tests

- [ ] **Step 5: Refactor `eurosat.py` onto the shared implementation**

In `src/eo_visual_retrieval/benchmarks/eurosat.py`, add to the imports (after the `datasets.eurosat` import block):

```python
from eo_visual_retrieval.benchmarks.coverage import EARTH_RADIUS_M, nearest_distances_m
```

Delete the module-level constant assignment `EARTH_RADIUS_M = 6_371_008.8` (line 45) — it is now imported.

Delete `_distances_m` entirely (lines 219-229) and replace `_minimum_distance` (lines 232-241) with:

```python
def _lonlat(candidates: list[EuroSatCandidate]) -> np.ndarray:
    return np.asarray([(item.longitude, item.latitude) for item in candidates], dtype=np.float64)


def _minimum_distance(
    left: list[EuroSatCandidate], right: list[EuroSatCandidate]
) -> float:
    if not left or not right:
        return math.inf
    return float(np.min(nearest_distances_m(_lonlat(left), _lonlat(right))))
```

At the former line 330, inside `select_spatial_split`, replace:

```python
        if float(np.min(_distances_m(candidate, query_lonlat))) < minimum_separation_m:
```

with:

```python
        candidate_lonlat = np.asarray([(candidate.longitude, candidate.latitude)], dtype=np.float64)
        if float(nearest_distances_m(candidate_lonlat, query_lonlat)[0]) < minimum_separation_m:
```

- [ ] **Step 6: Verify the refactor changed no behaviour**

Run the existing suite plus lint and types:

```bash
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m ruff check .
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m mypy
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest -q
```

Expected: Ruff clean, Mypy clean, all tests pass including the existing
`tests/test_eurosat.py::test_spatial_split_is_balanced_deterministic_and_separated`.

- [ ] **Step 7: Verify against the real published split**

This is the regression gate. The audit recomputes the guard band with the refactored code and must
reproduce the published values exactly.

```bash
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m eo_visual_retrieval.cli benchmark-eurosat-audit --manifest data/eurosat-v1/manifest.jsonl --image-root data/eurosat-v1/images
```

Expected, verbatim, in the JSON output:
- `"manifest_sha256": "bc0b10bf3e3cf29d7f7732529ce5f419b514e2ded3a5e2a5e6e88ebcdea45338"`
- `"minimum_separation_km": 5.066229991251209`
- `"items": 2000`, `"index": 1600`, `"query": 400`, `"spatial_groups": 725`, `"verified_files": 2000`

If `minimum_separation_km` differs at all, stop and investigate before continuing.

- [ ] **Step 8: Commit**

```bash
git add src/eo_visual_retrieval/benchmarks/coverage.py src/eo_visual_retrieval/benchmarks/eurosat.py tests/test_coverage.py
git commit -m "$(cat <<'EOF'
Give benchmark geography one great-circle implementation

Split preparation and split auditing each measured distance with the same
private haversine copy. Move a vectorised version into benchmarks/coverage.py
so a guard band cannot be audited by different arithmetic than enforced it.

Verified against the published EuroSAT v1 manifest: the audit still reproduces
5.066229991251209 km minimum separation and manifest SHA-256 bc0b10bf...45338.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: The two measurements

Adds the cell-budget and distance-tier measurements that the ADR's claims rest on.

**Files:**
- Modify: `src/eo_visual_retrieval/benchmarks/coverage.py`
- Test: `tests/test_coverage.py`

**Interfaces:**
- Consumes: `nearest_distances_m` from Task 1.
- Produces:
  - `CellBudget` frozen dataclass with fields `total_patches: int`, `total_cells: int`, `used_patches: int`, `used_cells: int`, `free_cells: int`, `free_patches: int`, `per_label: dict[str, dict[str, int]]`, and method `to_dict() -> dict[str, Any]`.
  - `cell_budget(candidates: Sequence[EuroSatCandidate], used_members: set[str]) -> CellBudget`
  - `distance_tiers(candidates: Sequence[EuroSatCandidate], used_members: set[str], reference_lonlat: NDArray[np.float64], thresholds_km: Sequence[float]) -> dict[str, Any]`

`EuroSatCandidate` is the existing frozen dataclass in `benchmarks/eurosat.py`; it exposes `member`, `label`, `longitude`, `latitude`, and `spatial_group`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_coverage.py`:

```python
from eo_visual_retrieval.benchmarks.coverage import cell_budget, distance_tiers
from eo_visual_retrieval.benchmarks.eurosat import EuroSatCandidate


def _candidate(label: str, member: str, cell: str, lon: float, lat: float) -> EuroSatCandidate:
    return EuroSatCandidate(
        member=member,
        label=label,
        source_crs="EPSG:32633",
        source_bounds=(0.0, 0.0, 640.0, 640.0),
        longitude=lon,
        latitude=lat,
        equal_area_x=0.0,
        equal_area_y=0.0,
        spatial_group=cell,
    )


def _sample() -> list[EuroSatCandidate]:
    return [
        _candidate("Forest", "a.tif", "cell-1", 0.0, 0.0),
        _candidate("Forest", "b.tif", "cell-1", 0.0, 0.1),
        _candidate("Forest", "c.tif", "cell-2", 0.0, 5.0),
        _candidate("River", "d.tif", "cell-2", 0.0, 5.1),
        _candidate("River", "e.tif", "cell-3", 0.0, 9.0),
    ]


def test_cell_budget_counts_cells_a_prepared_split_consumed() -> None:
    budget = cell_budget(_sample(), used_members={"a.tif"})

    assert budget.total_patches == 5
    assert budget.total_cells == 3
    assert budget.used_patches == 1
    assert budget.used_cells == 1
    assert budget.free_cells == 2
    # b.tif shares cell-1 with the used patch, so it is not free.
    assert budget.free_patches == 3


def test_cell_budget_reports_a_label_with_no_untouched_geography() -> None:
    """The finding that killed the EuroSAT holdout: a class can reach zero."""
    budget = cell_budget(_sample(), used_members={"c.tif", "e.tif"})

    assert budget.per_label["River"] == {"free_patches": 0, "free_cells": 0}
    assert budget.per_label["Forest"]["free_patches"] == 2


def test_cell_budget_rejects_a_member_that_is_not_a_candidate() -> None:
    with pytest.raises(ValueError, match="not present among the candidates"):
        cell_budget(_sample(), used_members={"absent.tif"})


def test_distance_tiers_shrink_as_the_threshold_grows() -> None:
    reference = np.asarray([[0.0, 0.0]])

    tiers = distance_tiers(
        _sample(), used_members={"a.tif"}, reference_lonlat=reference, thresholds_km=[1, 600]
    )

    # Every unused patch is at least 1 km from the reference point.
    assert tiers["1km"]["total"] == 4
    # Only patches beyond ~600 km survive: those near 5 and 9 degrees latitude.
    assert tiers["600km"]["total"] < tiers["1km"]["total"]
    assert tiers["1km"]["per_label"]["Forest"]["patches"] >= 1


def test_distance_tiers_report_the_smallest_label_and_label_count() -> None:
    reference = np.asarray([[0.0, 0.0]])

    tiers = distance_tiers(
        _sample(), used_members=set(), reference_lonlat=reference, thresholds_km=[1]
    )

    assert tiers["1km"]["labels_present"] == 2
    assert tiers["1km"]["min_label_patches"] == min(
        tiers["1km"]["per_label"][label]["patches"] for label in tiers["1km"]["per_label"]
    )


def test_distance_tiers_reject_a_non_positive_threshold() -> None:
    with pytest.raises(ValueError, match="thresholds_km must be positive"):
        distance_tiers(
            _sample(),
            used_members=set(),
            reference_lonlat=np.asarray([[0.0, 0.0]]),
            thresholds_km=[0],
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest tests/test_coverage.py -q`
Expected: FAIL — `ImportError: cannot import name 'cell_budget'`

- [ ] **Step 3: Implement the measurements**

Append to `src/eo_visual_retrieval/benchmarks/coverage.py`. Add these imports at the top of the file, after `from __future__ import annotations`:

```python
import collections
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, types only
    from eo_visual_retrieval.benchmarks.eurosat import EuroSatCandidate
```

Then append:

```python
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
    if not thresholds_km or any(value <= 0 for value in thresholds_km):
        raise ValueError("thresholds_km must be positive")

    unused = [candidate for candidate in candidates if candidate.member not in used_members]
    if not unused:
        raise ValueError("every candidate was used; no unused patch remains to measure")
    unused_lonlat = np.asarray(
        [(candidate.longitude, candidate.latitude) for candidate in unused], dtype=np.float64
    )
    nearest = nearest_distances_m(unused_lonlat, reference_lonlat)

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
            for label in sorted(patches)
        }
        tiers[f"{threshold:g}km"] = {
            "total": int(keep.sum()),
            "labels_present": len(per_label),
            "min_label_patches": min(
                (value["patches"] for value in per_label.values()), default=0
            ),
            "per_label": per_label,
        }
    return tiers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest tests/test_coverage.py -q`
Expected: PASS, 14 tests

- [ ] **Step 5: Run the full gates**

```bash
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m ruff check .
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m mypy
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest -q --cov=eo_visual_retrieval --cov-report=term --cov-fail-under=75
```

Expected: all clean, coverage at or above 75%.

- [ ] **Step 6: Commit**

```bash
git add src/eo_visual_retrieval/benchmarks/coverage.py tests/test_coverage.py
git commit -m "$(cat <<'EOF'
Measure how much dataset geography a benchmark has consumed

cell_budget reports the cells a prepared split spent and what remains per
label; distance_tiers reports the weaker distance-based fallback. Both are
needed to answer whether a dataset can still supply an untouched holdout.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: The reproducible measurement script

The ADR's central claim must be re-runnable. This is the CLI that produces it.

**Files:**
- Create: `scripts/eurosat_cell_budget.py`

**Interfaces:**
- Consumes: `cell_budget`, `distance_tiers` from Task 2; `discover_candidates` and `EuroSatCandidate` from `benchmarks/eurosat.py`; `read_jsonl` from `manifests.py`.
- Produces: a JSON report on stdout and at `--output`. Task 4 and Task 5 copy numbers from that report.

- [ ] **Step 1: Write the script**

Create `scripts/eurosat_cell_budget.py`:

```python
"""How much untouched geography EuroSAT still has after a prepared benchmark.

Reads every patch in the official archive, compares it with an already prepared
manifest, and reports both the cell budget and the distance-tier fallback. This
reproduces the measurement behind ADR 0006. No download and no training occurs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eo_visual_retrieval.benchmarks.coverage import cell_budget, distance_tiers
from eo_visual_retrieval.benchmarks.eurosat import discover_candidates
from eo_visual_retrieval.hashing import file_sha256
from eo_visual_retrieval.manifests import read_jsonl

DEFAULT_THRESHOLDS_KM = (5.0, 10.0, 20.0, 30.0, 50.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group-size-km", type=float, default=50.0)
    parser.add_argument(
        "--thresholds-km", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS_KM)
    )
    args = parser.parse_args()

    prepared = read_jsonl(args.manifest)
    used_members = {str(record.metadata["archive_member"]) for record in prepared}
    used_lonlat = np.asarray(
        [record.metadata["centroid_lonlat"] for record in prepared], dtype=np.float64
    )

    candidates = discover_candidates(args.archive, group_size_m=args.group_size_km * 1000)
    budget = cell_budget(candidates, used_members)
    tiers = distance_tiers(
        candidates,
        used_members=used_members,
        reference_lonlat=used_lonlat,
        thresholds_km=args.thresholds_km,
    )
    unused_lonlat_count = budget.total_patches - budget.used_patches

    result: dict[str, Any] = {
        "measurement": "eurosat-cell-budget-and-distance-tiers",
        "group_size_km": args.group_size_km,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "prepared_patches": len(prepared),
        "unused_patches": unused_lonlat_count,
        "cell_budget": budget.to_dict(),
        "distance_from_prepared": tiers,
        "notes": [
            "A cell counts as spent when any one of its patches was selected.",
            "Distance tiers are a weaker fallback than cell disjointness.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify lint and types accept it**

```bash
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m ruff check .
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m mypy
```

Expected: both clean. Mypy covers `scripts/` per `pyproject.toml`.

- [ ] **Step 3: Run it against the real archive**

This reads 27,000 patches and takes several minutes.

```bash
C:\Users\nikos\.venvs\eovr\Scripts\python.exe scripts/eurosat_cell_budget.py --archive data/downloads/EuroSAT_MS.zip --manifest data/eurosat-v1/manifest.jsonl --output outputs/eurosat-cell-budget.json
```

Expected values — these are the ADR's claims, so any mismatch must be investigated before writing it:

| Field | Expected |
|---|---:|
| `cell_budget.total_patches` | 27000 |
| `cell_budget.total_cells` | 845 |
| `cell_budget.used_cells` | 725 |
| `cell_budget.free_cells` | 120 |
| `cell_budget.free_patches` | 778 |
| `cell_budget.per_label.HerbaceousVegetation.free_patches` | 0 |
| `cell_budget.per_label.PermanentCrop.free_patches` | 1 |
| `cell_budget.per_label.AnnualCrop.free_patches` | 4 |
| `distance_from_prepared.10km.total` | 8445 |
| `distance_from_prepared.10km.min_label_patches` | 250 |
| `distance_from_prepared.50km.labels_present` | 5 |

`outputs/` is gitignored; the JSON stays local and its numbers are transcribed into the docs.

- [ ] **Step 4: Commit**

```bash
git add scripts/eurosat_cell_budget.py
git commit -m "$(cat <<'EOF'
Add a script reproducing the EuroSAT geography measurement

ADR 0006 rests on the claim that EuroSAT has no untouched geography left. This
re-runs that measurement from the official archive and a prepared manifest, so
the claim can be checked rather than trusted.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: ADR 0006

**Files:**
- Create: `docs/decisions/0006-confirmatory-evaluation-data.md`

**Interfaces:**
- Consumes: the JSON report from Task 3.
- Produces: the decision that Task 5's roadmap and validation entries reference.

- [ ] **Step 1: Write the ADR**

Create `docs/decisions/0006-confirmatory-evaluation-data.md`. Follow the shape of ADR 0005: front matter, Context, Decision, Options considered, Consequences, Action items, References.

```markdown
# ADR 0006: BigEarthNet v2 as the confirmatory evaluation set

- Status: accepted
- Date: 2026-09-03
- Decision owners: project maintainers

## Context

ADR 0005 required genuinely new, geographically separated data before confirmatory model
selection, and named BigEarthNet v2 as the likely source. It did not establish where an untouched
partition would come from. The cheaper plan was a holdout drawn from the 25,000 EuroSAT patches
v1 never selected.

Measurement rejected that plan. EuroSAT's 27,000 patches occupy only 845 distinct 50 km
equal-area cells, and preparing v1 consumed 725 of them.

| Quantity | Value |
|---|---:|
| Total patches | 27,000 |
| Total 50 km cells | 845 |
| Cells used by v1 | 725 |
| Cells untouched | 120 |
| Patches in untouched cells | 778 |

In those untouched cells, `HerbaceousVegetation` has zero patches, `PermanentCrop` has one, and
`AnnualCrop` has four. A class-balanced holdout is impossible, let alone the three mutually
disjoint partitions the protocol requires.

The cause is v1's own sampling: `_spread_sample` uses every available spatial group before reusing
one. That maximised v1's geographic diversity and spent the cell budget doing it. It was the right
choice for v1 and is not a defect.

Relaxing cell disjointness to a distance rule does not rescue the plan.

| Minimum distance from every v1 patch | Patches | Classes present | Smallest class |
|---|---:|---:|---:|
| 5 km | 16,024 | 10 | 1,094 |
| 10 km | 8,445 | 10 | 250 |
| 20 km | 1,473 | 10 | 12 |
| 30 km | 355 | 10 | 1 |
| 50 km | 65 | 5 | 0 |

A 10 km set is constructible before internal partition separation is subtracted, but its guarantee
is far weaker than v1's disjoint 50 km cells plus 5 km guard band, and it still cannot answer
whether SSL4EO's advantage is specific to EuroSAT.

Reproduce with `scripts/eurosat_cell_budget.py`.

## Decision

1. EuroSAT v1 is permanently a regression and development benchmark. No confirmatory claim rests
   on EuroSAT data.
2. BigEarthNet v2 (reBEN) is the single confirmatory evaluation set.
3. Relevance over its 19 CORINE labels is set similarity, not equality. Precision@k, Recall@k, and
   mAP@k are binary at `Jaccard >= 0.5`; nDCG@k uses the raw Jaccard as graded gain. The threshold
   is pre-registered. Sensitivity at 0.3, 0.5, and 0.7 is reported on the development partition
   only; the final partition is scored once at 0.5.
4. Three pre-registered partitions — 4,000 index, 500 development queries, 500 final queries — are
   drawn inside reBEN's official geographically separated splits and then independently audited
   with this project's own cell and guard-band machinery.
5. Acquisition is bounded. The distribution size, licence, and DOI are recorded before download;
   the archive is checksum-verified under `data/downloads/`; selected members are read directly
   from it rather than materialised as a duplicate dataset.
6. SSL4EO-S12 enters the confirmatory comparison only through a pre-registered gate, below.
7. The existing single-label evaluator path is not modified, so published EuroSAT results stay
   reproducible.

### The SSL4EO gate

BigEarthNet is 12-band Level-2A; sen2cor drops B10. The selected SSL4EO-S12 reference consumes
13-band Level-1C. The mismatch is both band count and radiometric quantity.

TerraMind registers `untok_sen2l2a@224` at 12 bands alongside `untok_sen2l1c@224` at 13, verified
against the installed TerraTorch, so it transfers to L2A natively. PCA and DINOv2 consume RGB and
are unaffected.

- If a published SSL4EO-S12 L2A ResNet-50 checkpoint exists in the SSL4EO-S12 repository, the
  TorchGeo weight registry, or the Hugging Face `torchgeo` organisation, and can be pinned by
  SHA-256, SSL4EO enters the comparison on L2A.
- Otherwise SSL4EO is recorded as absent from it, and EuroSAT v1 remains its only evidence.

Slicing the 13-channel `conv1` to 12 bands is rejected: it would silently alter a frozen
pretrained model, violate the frozen-encoder boundary, and produce scores not comparable with the
published EuroSAT numbers.

## Options considered

| Option | Evidence strength | Cost | Assessment |
|---|---|---|---|
| EuroSAT holdout from untouched cells | Would test new geography | None | **Impossible.** One class has zero patches available. |
| EuroSAT holdout at 10 km separation | Weak; same dataset and taxonomy | None | Rejected. Invites overclaiming from a guarantee weaker than v1's own. |
| BigEarthNet v2 | Cross-dataset, multi-label, timestamped | Download plus evaluator work | **Selected.** |
| Re-derive v1 with smaller cells | Would free cells | Invalidates published v1 results | Rejected. Re-opens ADR 0002 and discards executed evidence. |
| Another dataset, e.g. So2Sat LCZ42 | Single-label, city-separated | Research plus download | Deferred. Reconsider only if BigEarthNet acquisition proves unviable. |

## Consequences

- The project gains its first control for season and acquisition date. EuroSAT exposes no
  timestamps, a limitation recorded in `docs/validation.md`.
- Multi-label relevance is new machinery with a judgement in it. `tau = 0.5` is not derived from
  anything; it is pre-registered so it cannot be chosen after seeing scores.
- If the SSL4EO gate resolves to absent, the confirmatory comparison weakens to TerraMind against
  the RGB baselines, and every report of it must say so.
- Overlap between BigEarthNet and the pretraining corpora of the frozen encoders is a separate
  audit, unresolved here.
- Nothing in this ADR produces a score. It specifies data and protocol only.

## Action items

- [ ] Measure the reBEN distribution size and whether shard-level download is possible; record
      size, licence, and DOI before downloading.
- [ ] Resolve the SSL4EO L2A checkpoint gate to present-with-SHA or absent.
- [ ] Implement multi-label relevance as a separate path beside the single-label evaluator.
- [ ] Prepare and audit the three partitions; publish the achieved label distribution.
- [ ] Only then run the development comparison, freeze configuration, and score the final set once.

## Primary references

- [BigEarthNet](https://bigearth.net/)
- [reBEN: Refined BigEarthNet dataset](https://arxiv.org/pdf/2407.03653)
- [SSL4EO-S12](https://arxiv.org/pdf/2211.07044)
- [ADR 0002](0002-georeferenced-eurosat-benchmark.md), [ADR 0005](0005-evaluation-foundations-before-product.md)
```

- [ ] **Step 2: Verify every number against the JSON**

Open `outputs/eurosat-cell-budget.json` and confirm each figure in both ADR tables matches. The
published-hash test also runs over `docs/results/`, not `docs/decisions/`, so this check is manual
and must actually be done.

Run: `C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest -q`
Expected: PASS — no test should have changed.

- [ ] **Step 3: Commit**

```bash
git add docs/decisions/0006-confirmatory-evaluation-data.md
git commit -m "$(cat <<'EOF'
Add ADR 0006 selecting BigEarthNet v2 as the confirmatory set

EuroSAT cannot supply the untouched holdout ADR 0005 assumed: v1 consumed 725
of its 845 fifty-km cells, leaving one class with zero patches. Records the
measurement, makes EuroSAT v1 permanently a regression benchmark, and specifies
BigEarthNet v2 with Jaccard relevance at a pre-registered 0.5, three audited
partitions, and a gate for SSL4EO, which cannot read 12-band L2A patches.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Evidence and roadmap

**Files:**
- Modify: `docs/validation.md`
- Modify: `docs/project-context.md`
- Modify: `README.md:236`

**Interfaces:**
- Consumes: the ADR from Task 4 and the JSON from Task 3.
- Produces: nothing later tasks depend on. This is the final task.

- [ ] **Step 1: Add the validation entry**

In `docs/validation.md`, insert this section immediately before the line
`## Dependency vulnerability review — 2026-09-03`:

```markdown
## EuroSAT geography exhaustion — 2026-09-03

Executed with `scripts/eurosat_cell_budget.py` against the verified local archive and the
published v1 manifest, to test whether EuroSAT could supply the untouched holdout ADR 0005
assumed. It cannot.

| Gate | Executed evidence | Status |
|---|---|---|
| Source discovery | 27,000 georeferenced patches across 845 distinct 50 km EPSG:6933 cells | Passed |
| Cells consumed by v1 | 725 of 845, 86% | Measured |
| Untouched remainder | 120 cells holding 778 patches | Measured |
| Class availability | `HerbaceousVegetation` 0, `PermanentCrop` 1, `AnnualCrop` 4 | Blocking |
| Distance fallback | 8,445 patches at 10 km, smallest class 250; 65 patches and 5 classes at 50 km | Measured |
| Split regression | Audit still reproduces 5.066229991251209 km and manifest SHA-256 `bc0b10bf3e3cf29d7f7732529ce5f419b514e2ded3a5e2a5e6e88ebcdea45338` after the distance refactor | Passed |

The median unused patch lies 7.0 km from a v1 patch. A class-balanced holdout in untouched cells is
impossible, and the 10 km fallback carries a weaker guarantee than v1's own disjoint cells and
5 km guard band.

**Consequence:** EuroSAT v1 is permanently a regression and development benchmark. This measurement
supports no claim about BigEarthNet, model quality, or generalization; it establishes only that one
planned source of confirmatory data does not exist. See
[ADR 0006](decisions/0006-confirmatory-evaluation-data.md).
```

Then, in the `## Evidence not yet available` list, replace the line:

```markdown
- Retrieval quality for genuinely unseen query images; the new-image path is verified for
  numerical agreement only.
```

with:

```markdown
- Retrieval quality for genuinely unseen query images; the new-image path is verified for
  numerical agreement only.
- Any confirmatory result. No untouched evaluation partition exists yet; ADR 0006 specifies how
  one will be built, and nothing has been acquired or prepared.
```

- [ ] **Step 2: Update the roadmap in `docs/project-context.md`**

Replace the entire `## Next task` section with:

```markdown
## Next task

The evaluation-foundations gates are complete. The next phase was blocked on new held-out data,
and measurement has now settled where that data can come from: not EuroSAT. Preparing v1 consumed
725 of the dataset's 845 fifty-kilometre cells, leaving one class with no untouched patches at all.
EuroSAT v1 is therefore permanently a regression benchmark. See
[ADR 0006](decisions/0006-confirmatory-evaluation-data.md) and the measurement in
[validation](validation.md).

BigEarthNet v2 is the specified confirmatory set. In order:

1. Measure the reBEN distribution size and record its licence and DOI before downloading anything.
2. Resolve the SSL4EO L2A checkpoint gate; the current 13-band L1C reference cannot read
   BigEarthNet's 12-band L2A patches.
3. Implement multi-label Jaccard relevance beside the existing single-label evaluator, leaving
   published EuroSAT results reproducible.
4. Prepare and audit the three partitions, then tune on development queries only.
5. Test a Qdrant adapter against exact search, then return to the product surface.

Paid services, gated-model accounts, and distributed Milvus deployment remain deferred.
```

- [ ] **Step 3: Update the roadmap paragraph in `README.md`**

Replace the paragraph beginning `The locked-environment, local-tracking, GPU-parity` (around line
234) with:

```markdown
The locked-environment, local-tracking, GPU-parity, and frozen TerraMind regression gates have now
executed. Measurement then established that EuroSAT cannot supply a confirmatory holdout: preparing
v1 consumed 725 of its 845 fifty-kilometre cells, leaving one class with no untouched patches.
EuroSAT v1 is permanently a regression benchmark, and BigEarthNet v2 is the specified confirmatory
set. Exact search remains the default; Qdrant is the first future product-store experiment, and
Milvus is deferred until scale evidence justifies it. See
[ADR 0005](docs/decisions/0005-evaluation-foundations-before-product.md) and
[ADR 0006](docs/decisions/0006-confirmatory-evaluation-data.md).
```

- [ ] **Step 4: Run the full gates**

```bash
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m ruff check .
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m mypy
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest -q --cov=eo_visual_retrieval --cov-report=term --cov-fail-under=75
```

Expected: all clean. `tests/test_evidence.py` still passes because no file under `docs/results/`
was touched.

- [ ] **Step 5: Confirm published results are untouched**

```bash
git status --short docs/results/
```

Expected: empty output.

- [ ] **Step 6: Commit**

```bash
git add docs/validation.md docs/project-context.md README.md
git commit -m "$(cat <<'EOF'
Record the EuroSAT exhaustion measurement and re-point the roadmap

Adds the executed evidence for ADR 0006 under the evidence policy, states
plainly that no confirmatory result exists yet, and replaces the next-task list
with the BigEarthNet sequence: size and licence first, then the SSL4EO L2A
checkpoint gate, then multi-label relevance, then partitions.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: the exhaustion finding and its two tables to
Tasks 2-3 and the ADR in Task 4; relevance, partitions, the SSL4EO gate, and acquisition to the
ADR's Decision section in Task 4; all four deliverables in the spec's table to Tasks 3-5; the
acceptance criteria to Task 1 Step 7, Task 3 Step 3, Task 4 Step 2, and Task 5 Steps 4-5.

**Out-of-scope items stay out.** No task downloads BigEarthNet, implements the multi-label
evaluator, or prepares a partition. Those appear only as ADR action items.

**Type consistency.** `nearest_distances_m` takes `(left_lonlat, right_lonlat)` positionally in
Task 1 and is called that way in Tasks 1 and 2. `cell_budget(candidates, used_members)` and
`distance_tiers(candidates, used_members=..., reference_lonlat=..., thresholds_km=...)` match
between Task 2's definitions, Task 2's tests, and Task 3's call site. `CellBudget.to_dict()` is
defined in Task 2 and used in Task 3. `EARTH_RADIUS_M` moves from `eurosat.py` to `coverage.py` in
Task 1 Step 5 and is imported back, so its one remaining definition lives in `coverage.py`.

**Risk noted.** Task 1 changes code that produced published results. Step 7 is the gate: if the
audit's minimum separation shifts by any amount, stop rather than continue.
