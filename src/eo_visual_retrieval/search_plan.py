"""Transparent prompt defaults and strict metadata constraints, without an LLM."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

from eo_visual_retrieval.models import ImageRecord


@dataclass(frozen=True)
class SearchFilters:
    bbox: tuple[float, float, float, float] | None = None
    start_date: date | None = None
    end_date: date | None = None
    max_cloud_cover: float | None = None
    collection: str | None = None

    def __post_init__(self) -> None:
        if self.bbox is not None:
            if len(self.bbox) != 4 or not all(math.isfinite(v) for v in self.bbox):
                raise ValueError("bbox requires four finite WGS84 coordinates")
            west, south, east, north = self.bbox
            if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                raise ValueError(
                    "bbox must be west south east north; dateline crossing is unsupported"
                )
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start date must not follow end date")
        if self.max_cloud_cover is not None and not (
            math.isfinite(self.max_cloud_cover) and 0 <= self.max_cloud_cover <= 100
        ):
            raise ValueError("cloud coverage must be between 0 and 100 percent")
        if self.collection is not None and not re.fullmatch(r"[\w.-]{1,100}", self.collection):
            raise ValueError("collection must be a stable collection identifier")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "max_cloud_cover": self.max_cloud_cover,
            "collection": self.collection,
        }


@dataclass(frozen=True)
class SearchPlan:
    text: str
    filters: SearchFilters = field(default_factory=SearchFilters)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "filters": self.filters.to_dict(), "notes": self.notes}


def plan_query(
    text: str,
    *,
    overrides: SearchFilters | None = None,
    today: date | None = None,
    interpret: bool = True,
) -> SearchPlan:
    """Apply a small documented vocabulary, displaying every assumption.

    This is a convenience parser, not a general geocoder or natural-language
    constraint solver. Explicit controls always win. Users can disable defaults.
    """
    text = text.strip()
    if len(text) > 1000:
        raise ValueError("text must contain at most 1000 characters")
    today = today or datetime.now(UTC).date()
    filters = SearchFilters()
    notes = [
        "Search covers the local index only; it does not discover or download new scenes.",
        "Place, date, sensor and cloud requirements are guaranteed only by the displayed filters. "
        "Use explicit controls for constraints outside the supported prompt defaults.",
    ]
    if interpret and text:
        if re.search(r"\b(?:not|except|excluding|without)\b", text, re.I):
            notes.append("Negation detected: automatic defaults disabled; set explicit filters.")
        else:
            if re.search(r"\bnear Athens\b", text, re.I):
                filters = replace(filters, bbox=(23.4, 37.7, 24.1, 38.2))
                notes.append("'Near Athens' uses an approximate Athens, Greece bounding box.")
            if re.search(r"\brecent\b", text, re.I):
                filters = replace(filters, start_date=today - timedelta(days=90), end_date=today)
                notes.append("'Recent' means the last 90 days, anchored to today's UTC date.")
            if re.search(r"\blow cloud (?:coverage|cover)\b", text, re.I):
                filters = replace(filters, max_cloud_cover=10.0)
                notes.append("'Low cloud coverage' means scene cloud cover at most 10%.")
            if re.search(r"\bSentinel(?:[- ]2)? imagery\b", text, re.I):
                filters = replace(filters, collection="sentinel-2-l2a")
                notes.append("'Sentinel imagery' defaults to the sentinel-2-l2a collection.")
    if overrides is not None:
        # Construct once: replacing one endpoint at a time could reject a valid
        # explicit interval against the old inferred endpoint.
        values = {
            key: getattr(overrides, key)
            if getattr(overrides, key) is not None
            else getattr(filters, key)
            for key in ("bbox", "start_date", "end_date", "max_cloud_cover", "collection")
        }
        filters = SearchFilters(**values)
        if any(value is not None for value in overrides.to_dict().values()):
            notes.append(
                "Explicit controls override prompt defaults; the applied filters are below."
            )
    if re.search(r"\b(expansion|change|growth|deforestation)\b", text, re.I):
        notes.append(
            "Single-scene similarity cannot verify change. Expansion needs aligned observations "
            "from multiple dates and a separate change assessment."
        )
    if filters.max_cloud_cover is not None:
        notes.append("Scene cloud percentage does not guarantee a cloud-free chip.")
    return SearchPlan(text=text, filters=filters, notes=tuple(notes))


def scene_metadata(record: ImageRecord) -> dict[str, Any]:
    """Expose only safe search facts, never arbitrary source metadata or paths."""
    source = record.metadata
    raw_center = source.get("centroid_lonlat")
    if raw_center is None and source.get("bbox_crs") == "EPSG:4326":
        bbox = source.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                raw_center = [
                    (float(bbox[0]) + float(bbox[2])) / 2,
                    (float(bbox[1]) + float(bbox[3])) / 2,
                ]
            except (ValueError, TypeError):
                raw_center = None
    center = None
    if isinstance(raw_center, (list, tuple)) and len(raw_center) == 2:
        try:
            lon, lat = float(raw_center[0]), float(raw_center[1])
            if (
                math.isfinite(lon)
                and math.isfinite(lat)
                and -180 <= lon <= 180
                and -90 <= lat <= 90
            ):
                center = (lon, lat)
        except (ValueError, TypeError):
            pass
    cloud = source.get("eo_cloud_cover", source.get("eo:cloud_cover"))
    if isinstance(cloud, bool) or not isinstance(cloud, (float, int)) or not 0 <= cloud <= 100:
        cloud = None
    acquired = None
    try:
        raw_date = source.get("datetime")
        if isinstance(raw_date, str):
            timestamp = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            if timestamp.tzinfo is not None:
                acquired = timestamp.astimezone(UTC).date().isoformat()
    except ValueError:
        pass
    collection = source.get("collection")
    if not isinstance(collection, str) or not re.fullmatch(r"[\w.-]{1,100}", collection):
        collection = None
    return {
        "centroid_lonlat": center,
        "date": acquired,
        "cloud_cover": cloud,
        "collection": collection,
    }


def matches_filters(metadata: dict[str, Any], filters: SearchFilters) -> bool:
    """Missing or malformed metadata never satisfies an active constraint."""
    return all(value == "pass" for value in filter_checks(metadata, filters).values())


def filter_checks(metadata: dict[str, Any], filters: SearchFilters) -> dict[str, str]:
    """The same decisions drive eligibility and its user-visible explanation."""
    checks: dict[str, str] = {}

    def check(key: str, known: bool, passed: bool) -> None:
        checks[key] = "missing" if not known else "pass" if passed else "fail"

    if filters.collection:
        check("collection", metadata["collection"] is not None,
              metadata["collection"] == filters.collection)
    if filters.max_cloud_cover is not None:
        cloud = metadata["cloud_cover"]
        check("max_cloud_cover", cloud is not None,
              cloud is not None and cloud <= filters.max_cloud_cover)
    if filters.start_date or filters.end_date:
        acquired = date.fromisoformat(metadata["date"]) if metadata["date"] else None
        if filters.start_date:
            check("start_date", acquired is not None,
                  acquired is not None and acquired >= filters.start_date)
        if filters.end_date:
            check("end_date", acquired is not None,
                  acquired is not None and acquired <= filters.end_date)
    if filters.bbox:
        center = metadata["centroid_lonlat"]
        west, south, east, north = filters.bbox
        check("bbox", center is not None, center is not None
              and west <= center[0] <= east and south <= center[1] <= north)
    return checks
