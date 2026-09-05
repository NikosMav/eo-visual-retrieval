"""Reduce published evidence files into the payload the findings page draws.

The findings served here are not written into the page. They are read from the
same JSON reports committed under ``docs/results/``, so a served figure and the
recorded evidence cannot disagree: re-run an evaluation and the page moves with
it, or the file is missing and the page is not offered at all.

Nothing here computes a metric. Every value is lifted from a report produced by
``eovr analyze-retrieval`` or ``eovr temporal-survey``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ANALYSIS_FILE = "eurosat-v1-analysis.json"
AVAILABILITY_FILE = "temporal-availability-2024.json"

# Store names carry their corpus and checkpoint; a chart axis needs a short one.
# Presentation only, and an unknown store keeps its recorded name rather than
# being silently relabelled.
DISPLAY_NAMES = {
    "eurosat-v1-pca-64": "PCA-64",
    "eurosat-v1-dinov2-vits14": "DINOv2",
    "eurosat-v1-ssl4eo-s12-rgb-moco-resnet50": "SSL4EO RGB",
    "eurosat-v1-ssl4eo-s12-moco-resnet50": "SSL4EO 13-band",
    "eurosat-v1-terramind-tiny": "TerraMind",
}

QUARTILES = ("q1", "q2", "q3", "q4")


def _display(store_name: str) -> str:
    return DISPLAY_NAMES.get(store_name, store_name)


def _place_label(place_id: str) -> str:
    """Turn a frozen place identifier into something readable on an axis.

    Identifiers name a region and then a local specifier, as in
    ``po-valley-cremona``. Dropping the last segment keeps the region, which is
    what an axis has room for; the full identifier stays in the tooltip.
    """

    parts = place_id.split("-")
    region = " ".join(parts[:-1]) if len(parts) > 1 else place_id
    return region[:1].upper() + region[1:]


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read evidence file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"evidence file is not a JSON object: {path}")
    return payload


def build_payload(results_dir: Path) -> dict[str, Any]:
    """Read the committed reports and reduce them for the page.

    Raises if a required section is absent rather than rendering a chart with
    holes in it: a findings page missing its findings is worse than no page.
    """

    analysis = _read(results_dir / ANALYSIS_FILE)
    availability = _read(results_dir / AVAILABILITY_FILE)
    return reduce_reports(analysis, availability)


def reduce_reports(analysis: dict[str, Any], availability: dict[str, Any]) -> dict[str, Any]:
    """Reduce already-snapshotted reports without re-reading mutable files."""

    for section in ("aggregate", "agreement", "geographic", "distance", "failures"):
        if section not in analysis:
            raise ValueError(f"{ANALYSIS_FILE} has no '{section}' section")

    spread = analysis["geographic"]["spatial_group_spread"]
    distance = analysis["distance"]["nearest_same_label_index_m"]

    series = []
    for store, aggregate in analysis["aggregate"].items():
        cells = spread[store]["group_mean_map_at_k"]
        quartiles = distance[store]["quartiles"]
        series.append(
            {
                "key": _display(store),
                "map": aggregate["map_at_k"],
                "sd": cells["standard_deviation"],
                "min": cells["minimum"],
                "med": cells["median"],
                "q": [quartiles[name]["map_at_k"] for name in QUARTILES],
            }
        )
    series.sort(key=lambda row: row["map"], reverse=True)

    overlap = {
        " vs ".join(_display(part) for part in pair.split(" vs ")): value[
            "mean_overlap_at_k"
        ]
        for pair, value in analysis["agreement"]["pairwise_overlap"].items()
    }

    edges = next(iter(distance.values()))["quartile_edges_m"]
    hard = analysis["failures"]["universally_hard_queries"]

    places = [
        {
            "label": _place_label(row["place_id"]),
            "place_id": row["place_id"],
            "latitude": row["latitude"],
            "days": row["distinct_days"],
        }
        for row in sorted(availability["results"], key=lambda row: row["latitude"])
    ]

    return {
        "k": analysis["k"],
        "queries": analysis["agreement"]["queries"],
        "series": series,
        "overlap": overlap,
        "overlap_max": max(overlap.values()) if overlap else 0.0,
        "top_one": analysis["agreement"]["top_one_correctness"],
        "quartile_edges_km": [round(edge / 1000, 1) for edge in edges],
        "universally_hard": {
            "count": hard["count"],
            "examples": [row["label"] for row in hard["examples"]],
        },
        "places": places,
        "availability": {
            "datetime": availability["datetime"],
            "max_cloud_cover": availability["max_cloud_cover"],
            "total_days": availability["distinct_days"]["total"],
            "collection": availability["collection"],
        },
    }


def load_findings(results_dir: Path | None) -> dict[str, Any] | None:
    """Return the payload, or ``None`` when this deployment serves no findings.

    Absent evidence disables the page. It is never replaced with placeholder
    numbers, because a plausible-looking figure that nothing produced is the one
    failure mode this whole project is arranged against.
    """

    if results_dir is None:
        return None
    if not results_dir.is_dir():
        raise ValueError(f"results directory does not exist: {results_dir}")
    return build_payload(results_dir)
