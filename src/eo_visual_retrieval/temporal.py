"""Places observed repeatedly, for retrieval judged by identity rather than class.

EuroSAT asks whether a retrieved image is the same *kind* of place. With
acquisition dates a sharper question becomes available: given one view of a
place, does the system retrieve that same place seen on another date? The answer
needs no class labels, and its ground truth cannot be argued with.

This module handles the discovery half. It asks the catalogue what exists over a
set of fixed locations across a date range, and reports how many usable dates
each place has, so a corpus is committed to only after its feasibility is
measured rather than assumed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from eo_visual_retrieval.models import StacItemRecord
from eo_visual_retrieval.stac import StacSearchConfig, search_stac

# One degree of latitude, close enough for choosing a search window. Real
# distances in this project are great-circle; see benchmarks/coverage.py.
METRES_PER_DEGREE_LATITUDE = 111_320.0

# A place needs at least this many dates to contribute a query and an answer.
MINIMUM_DATES_PER_PLACE = 2


@dataclass(frozen=True)
class Place:
    """A fixed ground location observed on many dates."""

    place_id: str
    longitude: float
    latitude: float
    note: str = ""

    def __post_init__(self) -> None:
        if not self.place_id or "/" in self.place_id or "\\" in self.place_id:
            raise ValueError(f"place_id must be a non-empty path-safe name: {self.place_id!r}")
        if not math.isfinite(self.longitude) or not -180 <= self.longitude <= 180:
            raise ValueError(f"longitude out of range for {self.place_id!r}: {self.longitude}")
        if not math.isfinite(self.latitude) or not -85 <= self.latitude <= 85:
            raise ValueError(f"latitude out of range for {self.place_id!r}: {self.latitude}")

    def bbox(self, *, size_m: float) -> tuple[float, float, float, float]:
        """Return a square-ish search window centred on this place.

        Longitude degrees shorten towards the poles, so the east-west extent is
        divided by the cosine of the latitude. This sizes a *search* window, not
        a measured distance, and the pixel grid is fixed later by the chip
        builder from the raster's own CRS.
        """

        if not math.isfinite(size_m) or size_m <= 0:
            raise ValueError("size_m must be a positive number of metres")
        half_lat = (size_m / 2) / METRES_PER_DEGREE_LATITUDE
        cosine = math.cos(math.radians(self.latitude))
        if cosine <= 0:
            raise ValueError(f"latitude too close to a pole for {self.place_id!r}")
        half_lon = half_lat / cosine
        west, east = self.longitude - half_lon, self.longitude + half_lon
        south, north = self.latitude - half_lat, self.latitude + half_lat
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError(f"window for {self.place_id!r} leaves valid coordinates")
        return (west, south, east, north)


# A frozen, auditable selection, kept in code for the same reason the EuroSAT
# classes and the BigEarthNet partitions are: a selection that can drift between
# runs is not a selection. Locations spread across the latitude range EuroSAT v1
# covers, because the structure analysis found quality varies by band and found
# 35-40N hardest for every representation. Each is a generic, public landscape;
# none is a private or sensitive area of interest.
EUROPE_LATITUDE_SPREAD_V1: tuple[dict[str, Any], ...] = (
    {"place_id": "crete-messara", "longitude": 24.95, "latitude": 35.05,
     "note": "35-40N; Mediterranean agricultural plain with strong dry-season contrast"},
    {"place_id": "attica-thriasio", "longitude": 23.58, "latitude": 38.53,
     "note": "35-40N; mixed peri-urban and agricultural land east of Athens"},
    {"place_id": "andalusia-guadalquivir", "longitude": -5.55, "latitude": 37.55,
     "note": "35-40N; irrigated river valley, the hardest latitude band on EuroSAT v1"},
    {"place_id": "po-valley-cremona", "longitude": 10.02, "latitude": 45.14,
     "note": "40-45N; intensive irrigated cropland with a pronounced growing season"},
    {"place_id": "beauce-orleans", "longitude": 1.85, "latitude": 48.05,
     "note": "45-50N; open-field cereal plain, near-total seasonal cover change"},
    {"place_id": "flevoland-polder", "longitude": 5.55, "latitude": 52.52,
     "note": "50-55N; reclaimed polder, geometric parcels and permanent water nearby"},
    {"place_id": "brandenburg-spreewald", "longitude": 13.98, "latitude": 51.88,
     "note": "50-55N; mixed forest, wetland and small-parcel agriculture"},
    {"place_id": "jutland-viborg", "longitude": 9.40, "latitude": 56.45,
     "note": "55-60N; northern mixed farmland with a short growing season"},
    {"place_id": "smaland-vaxjo", "longitude": 14.80, "latitude": 56.88,
     "note": "55-60N; boreal forest and lakes, heavy snow contrast in winter"},
    {"place_id": "uppland-uppsala", "longitude": 17.63, "latitude": 59.86,
     "note": "55-60N; forest and farmland mosaic at the edge of reliable optical coverage"},
    {"place_id": "ostrobothnia-vaasa", "longitude": 21.85, "latitude": 63.10,
     "note": "60-65N; high-latitude coast, low sun elevation and long winter snow"},
    {"place_id": "lapland-rovaniemi", "longitude": 25.72, "latitude": 66.50,
     "note": "above 65N; tests where optical retrieval stops working at all"},
)


def read_places(payload: Sequence[dict[str, Any]]) -> list[Place]:
    """Build places from a plain configuration list, rejecting duplicates."""

    places = []
    seen = set()
    for entry in payload:
        place = Place(
            place_id=str(entry["place_id"]),
            longitude=float(entry["longitude"]),
            latitude=float(entry["latitude"]),
            note=str(entry.get("note", "")),
        )
        if place.place_id in seen:
            raise ValueError(f"duplicate place_id: {place.place_id}")
        seen.add(place.place_id)
        places.append(place)
    if not places:
        raise ValueError("at least one place is required")
    return places


def _acquisition_day(record: StacItemRecord) -> str:
    return str(record.datetime)[:10] if record.datetime else "unknown"


def summarize_place(place: Place, records: Sequence[StacItemRecord]) -> dict[str, Any]:
    """Describe what one place offers, one entry per distinct acquisition day.

    Sentinel-2 splits a pass into tiles, so one place near a tile boundary can
    appear in several items from the same overpass. Those are one observation,
    not several, and counting them separately would overstate how many dates a
    place really has.
    """

    by_day: dict[str, StacItemRecord] = {}
    for record in records:
        day = _acquisition_day(record)
        existing = by_day.get(day)
        if existing is None or _cloud(record) < _cloud(existing):
            by_day[day] = record

    observations = [
        {
            "acquisition_day": day,
            "item_id": record.item_id,
            "datetime": record.datetime,
            "eo:cloud_cover": _cloud(record),
            "grid:code": record.properties.get("grid:code")
            or record.properties.get("s2:mgrs_tile"),
            "sat:relative_orbit": record.properties.get("sat:relative_orbit"),
            "view:sun_elevation": record.properties.get("view:sun_elevation"),
        }
        for day, record in sorted(by_day.items())
    ]
    days = [entry["acquisition_day"] for entry in observations]
    return {
        "place_id": place.place_id,
        "longitude": place.longitude,
        "latitude": place.latitude,
        "note": place.note,
        "items_returned": len(records),
        "distinct_days": len(observations),
        "usable": len(observations) >= MINIMUM_DATES_PER_PLACE,
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "observations": observations,
    }


def _cloud(record: StacItemRecord) -> float:
    value = record.properties.get("eo:cloud_cover")
    return float(value) if isinstance(value, (int, float)) else float("inf")


def survey_places(
    places: Sequence[Place],
    *,
    api_url: str,
    collection: str,
    datetime_range: str,
    window_m: float,
    max_cloud_cover: float | None,
    limit: int,
) -> dict[str, Any]:
    """Ask the catalogue what each place offers, without downloading pixels."""

    surveys = []
    for place in places:
        config = StacSearchConfig(
            api_url=api_url,
            collection=collection,
            bbox=place.bbox(size_m=window_m),
            datetime=datetime_range,
            limit=limit,
            max_cloud_cover=max_cloud_cover,
        )
        surveys.append(summarize_place(place, search_stac(config)))

    usable = [survey for survey in surveys if survey["usable"]]
    day_counts = [survey["distinct_days"] for survey in surveys]
    return {
        "survey": "temporal-place-availability",
        "api_url": api_url,
        "collection": collection,
        "datetime": datetime_range,
        "window_m": window_m,
        "max_cloud_cover": max_cloud_cover,
        "limit_per_place": limit,
        "places": len(surveys),
        "usable_places": len(usable),
        "distinct_days": {
            "minimum": min(day_counts) if day_counts else 0,
            "median": sorted(day_counts)[len(day_counts) // 2] if day_counts else 0,
            "maximum": max(day_counts) if day_counts else 0,
            "total": sum(day_counts),
        },
        "note": (
            "one entry per acquisition day; tiles from the same overpass are "
            "collapsed, keeping the least cloudy. No imagery was downloaded."
        ),
        "results": surveys,
    }
