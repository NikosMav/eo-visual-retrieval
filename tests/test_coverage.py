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
