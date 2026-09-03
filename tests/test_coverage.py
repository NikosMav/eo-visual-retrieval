"""Geographic coverage measurements shared by benchmark preparation and audits."""

from __future__ import annotations

import numpy as np
import pytest

from eo_visual_retrieval.benchmarks.coverage import (
    EARTH_RADIUS_M,
    cell_budget,
    distance_tiers,
    nearest_distance_percentiles,
    nearest_distances_m,
)
from eo_visual_retrieval.benchmarks.eurosat import EuroSatCandidate


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

    # a.tif sits on the reference point itself, so it falls below 1 km and is
    # dropped; the remaining two Forest and two River patches all survive.
    assert tiers["1km"]["total"] == 4
    assert tiers["1km"]["labels_present"] == 2
    assert tiers["1km"]["min_label_patches"] == 2
    assert tiers["1km"]["per_label"] == {
        "Forest": {"patches": 2, "cells": 2},
        "River": {"patches": 2, "cells": 2},
    }


def test_distance_tiers_reports_a_label_eliminated_at_a_threshold() -> None:
    """The finding that killed the EuroSAT holdout: a class can reach zero.

    At 600 km, both Forest patches (~556 km and ~11 km away) fall below the
    threshold, while the farther River patch (~1001 km away) survives. A
    reader must see Forest reported as empty, not silently dropped from the
    table.
    """
    reference = np.asarray([[0.0, 0.0]])

    tiers = distance_tiers(
        _sample(), used_members=set(), reference_lonlat=reference, thresholds_km=[600]
    )

    assert tiers["600km"]["per_label"]["Forest"] == {"patches": 0, "cells": 0}
    assert tiers["600km"]["min_label_patches"] == 0
    assert tiers["600km"]["labels_present"] == 1


def test_distance_tiers_reject_a_non_positive_threshold() -> None:
    with pytest.raises(ValueError, match="thresholds_km must be positive"):
        distance_tiers(
            _sample(),
            used_members=set(),
            reference_lonlat=np.asarray([[0.0, 0.0]]),
            thresholds_km=[0],
        )


def test_nearest_distance_percentiles_returns_one_key_per_percentile() -> None:
    percentiles = nearest_distance_percentiles(
        _sample(),
        used_members={"a.tif"},
        reference_lonlat=np.asarray([[0.0, 0.0]]),
    )

    assert set(percentiles) == {"p5", "p25", "p50", "p75", "p95"}


def test_nearest_distance_percentiles_matches_a_hand_computed_median() -> None:
    """b/c/d/e sit at 0.1/5.0/5.1/9.0 degrees latitude from the reference.

    One degree here is EARTH_RADIUS_M * radians(1) metres, so the four
    unused distances in km are 11.119..., 555.975..., 567.094..., 1000.754...
    and the median of four values is the mean of the middle two.
    """
    degree_km = EARTH_RADIUS_M * np.radians(1.0) / 1000
    expected_p50 = round((5.0 * degree_km + 5.1 * degree_km) / 2, 3)

    percentiles = nearest_distance_percentiles(
        _sample(),
        used_members={"a.tif"},
        reference_lonlat=np.asarray([[0.0, 0.0]]),
        percentiles=[50],
    )

    assert percentiles == {"p50": pytest.approx(expected_p50)}


def test_nearest_distance_percentiles_rejects_an_empty_candidate_list() -> None:
    with pytest.raises(ValueError, match="at least one candidate is required"):
        nearest_distance_percentiles(
            [], used_members=set(), reference_lonlat=np.asarray([[0.0, 0.0]])
        )


def test_nearest_distance_percentiles_rejects_an_empty_percentile_list() -> None:
    with pytest.raises(ValueError, match="percentiles must not be empty"):
        nearest_distance_percentiles(
            _sample(),
            used_members=set(),
            reference_lonlat=np.asarray([[0.0, 0.0]]),
            percentiles=[],
        )


def test_nearest_distance_percentiles_rejects_when_every_candidate_was_used() -> None:
    all_members = {candidate.member for candidate in _sample()}

    with pytest.raises(ValueError, match="every candidate was used; no unused patch remains"):
        nearest_distance_percentiles(
            _sample(),
            used_members=all_members,
            reference_lonlat=np.asarray([[0.0, 0.0]]),
        )
