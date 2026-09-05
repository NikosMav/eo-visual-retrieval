"""Place definitions and availability summaries for date-based retrieval."""

from __future__ import annotations

import math

import pytest

from eo_visual_retrieval.models import StacItemRecord
from eo_visual_retrieval.temporal import (
    EUROPE_LATITUDE_SPREAD_V1,
    MINIMUM_DATES_PER_PLACE,
    Place,
    read_places,
    summarize_place,
)


def _item(item_id: str, moment: str, cloud: float, tile: str = "MGRS-33UUB") -> StacItemRecord:
    return StacItemRecord(
        api_url="https://example.test/stac",
        collection="sentinel-2-l2a",
        item_id=item_id,
        bbox=None,
        datetime=moment,
        asset_keys=("B02",),
        properties={"eo:cloud_cover": cloud, "grid:code": tile, "sat:relative_orbit": 93},
    )


def test_place_window_narrows_with_latitude() -> None:
    """A degree of longitude shrinks towards the poles, so the window must widen."""

    south = Place(place_id="south", longitude=0.0, latitude=35.0)
    north = Place(place_id="north", longitude=0.0, latitude=65.0)

    south_west, _, south_east, _ = south.bbox(size_m=2560)
    north_west, _, north_east, _ = north.bbox(size_m=2560)

    assert (north_east - north_west) > (south_east - south_west)
    # The north-south extent does not depend on latitude.
    _, s_south, _, s_north = south.bbox(size_m=2560)
    _, n_south, _, n_north = north.bbox(size_m=2560)
    assert (s_north - s_south) == pytest.approx(n_north - n_south)


def test_place_window_is_centred_on_the_place() -> None:
    place = Place(place_id="centre", longitude=10.0, latitude=50.0)
    west, south, east, north = place.bbox(size_m=1000)

    assert (west + east) / 2 == pytest.approx(10.0)
    assert (south + north) / 2 == pytest.approx(50.0)
    assert (north - south) * 111_320.0 == pytest.approx(1000.0, rel=1e-6)


def test_places_reject_unusable_definitions() -> None:
    with pytest.raises(ValueError, match="path-safe"):
        Place(place_id="a/b", longitude=0.0, latitude=0.0)
    with pytest.raises(ValueError, match="longitude out of range"):
        Place(place_id="a", longitude=200.0, latitude=0.0)
    with pytest.raises(ValueError, match="latitude out of range"):
        Place(place_id="a", longitude=0.0, latitude=95.0)
    with pytest.raises(ValueError, match="positive number of metres"):
        Place(place_id="a", longitude=0.0, latitude=0.0).bbox(size_m=0)
    with pytest.raises(ValueError, match="positive number of metres"):
        Place(place_id="a", longitude=0.0, latitude=0.0).bbox(size_m=math.inf)


def test_duplicate_places_are_rejected() -> None:
    entry = {"place_id": "same", "longitude": 1.0, "latitude": 2.0}
    with pytest.raises(ValueError, match="duplicate place_id"):
        read_places([entry, dict(entry)])
    with pytest.raises(ValueError, match="at least one place"):
        read_places([])


def test_frozen_selection_is_valid_and_spans_the_latitude_range() -> None:
    places = read_places(list(EUROPE_LATITUDE_SPREAD_V1))

    assert len(places) == 12
    latitudes = [place.latitude for place in places]
    assert min(latitudes) < 36 and max(latitudes) > 65
    assert all(place.note for place in places), "every place records why it was chosen"


def test_one_overpass_split_across_tiles_counts_as_one_date() -> None:
    """Sentinel-2 tiles a pass, so a place on a boundary appears twice per overpass.

    Counting those separately would overstate how many dates a place offers, and
    would put two views of one moment on both sides of a query/index split.
    """

    place = Place(place_id="edge", longitude=10.0, latitude=50.0)
    records = [
        _item("scene_T33UUB", "2024-05-01T10:00:00Z", 8.0, tile="MGRS-33UUB"),
        _item("scene_T33UUC", "2024-05-01T10:00:02Z", 2.0, tile="MGRS-33UUC"),
        _item("scene_later", "2024-07-14T10:00:00Z", 1.0),
    ]
    summary = summarize_place(place, records)

    assert summary["items_returned"] == 3
    assert summary["distinct_days"] == 2
    # The retained view of the shared day is the least cloudy one.
    kept = [row for row in summary["observations"] if row["acquisition_day"] == "2024-05-01"]
    assert kept[0]["item_id"] == "scene_T33UUC"
    assert kept[0]["eo:cloud_cover"] == 2.0


def test_a_place_with_one_date_cannot_support_retrieval() -> None:
    """With a single view there is nothing to retrieve when it becomes the query."""

    place = Place(place_id="lonely", longitude=10.0, latitude=50.0)
    summary = summarize_place(place, [_item("only", "2024-05-01T10:00:00Z", 1.0)])

    assert summary["distinct_days"] == 1
    assert summary["usable"] is False
    assert MINIMUM_DATES_PER_PLACE == 2


def test_missing_cloud_cover_sorts_last_rather_than_crashing() -> None:
    place = Place(place_id="unknown-cloud", longitude=10.0, latitude=50.0)
    without = _item("no-cloud", "2024-05-01T10:00:00Z", 0.0)
    without = StacItemRecord(
        api_url=without.api_url,
        collection=without.collection,
        item_id="no-cloud",
        bbox=None,
        datetime="2024-05-01T10:00:00Z",
        asset_keys=without.asset_keys,
        properties={},
    )
    summary = summarize_place(place, [without, _item("with-cloud", "2024-05-01T10:30:00Z", 5.0)])

    assert summary["distinct_days"] == 1
    assert summary["observations"][0]["item_id"] == "with-cloud"
