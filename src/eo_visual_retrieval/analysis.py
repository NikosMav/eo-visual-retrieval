"""Structure behind the headline retrieval metrics.

An aggregate mAP@k says how well a representation ranked, not why. This module
re-reads the same per-query results the evaluator already produces and groups
them four ways: by geography, by how much the representations agree, by what
they retrieve when they are wrong, and by how far a query sits from anything it
could match. Each grouping answers a question the aggregate cannot.

Nothing here re-ranks or re-implements a metric. Every number is an aggregation
of :func:`eo_visual_retrieval.evaluation.evaluate_queries` output through
:func:`eo_visual_retrieval.evaluation.summarize_queries`, so a slice and the
published headline can never disagree about what a mean is.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eo_visual_retrieval.benchmarks.coverage import nearest_distances_m
from eo_visual_retrieval.evaluation import QueryEvaluation, summarize_queries
from eo_visual_retrieval.models import ImageRecord

# Reporting a mean over very few queries invites reading noise as a finding.
MINIMUM_SLICE_QUERIES = 5

# Enough failures to see a pattern without pasting a corpus into a report.
MAXIMUM_LISTED_QUERIES = 25


@dataclass(frozen=True)
class QueryGeography:
    """Where a query sits, and how far it is from anything it could match."""

    item_id: str
    label: str
    longitude: float
    latitude: float
    spatial_group: str
    nearest_index_m: float
    nearest_same_label_index_m: float


def _lonlat(records: Sequence[ImageRecord]) -> NDArray[np.float64]:
    points = []
    for record in records:
        centroid = record.metadata.get("centroid_lonlat")
        if centroid is None or len(centroid) != 2:
            raise ValueError(f"record {record.item_id!r} has no centroid_lonlat")
        points.append([float(centroid[0]), float(centroid[1])])
    return np.asarray(points, dtype=np.float64)


def query_geography(records: Sequence[ImageRecord]) -> dict[str, QueryGeography]:
    """Locate every query and measure its distance to the index partition.

    Two distances are recorded because they answer different questions. The
    nearest index item of any label describes how densely the corpus covers that
    place; the nearest index item of the *same* label describes whether a correct
    answer even exists nearby. A benchmark that only works where a same-label
    neighbour is metres away is measuring proximity, not representation quality.
    """

    queries = [record for record in records if record.split == "query"]
    indexes = [record for record in records if record.split == "index"]
    if not queries or not indexes:
        raise ValueError("records must contain both query and index items")

    nearest_any = nearest_distances_m(_lonlat(queries), _lonlat(indexes))

    nearest_same = np.full(len(queries), math.inf, dtype=np.float64)
    for label in sorted({record.label for record in queries if record.label is not None}):
        same_label_indexes = [record for record in indexes if record.label == label]
        positions = [i for i, record in enumerate(queries) if record.label == label]
        if not same_label_indexes or not positions:
            continue
        subset = [queries[position] for position in positions]
        distances = nearest_distances_m(_lonlat(subset), _lonlat(same_label_indexes))
        nearest_same[positions] = distances

    located = {}
    for position, record in enumerate(queries):
        centroid = record.metadata["centroid_lonlat"]
        located[record.item_id] = QueryGeography(
            item_id=record.item_id,
            label=record.label or "unlabeled",
            longitude=float(centroid[0]),
            latitude=float(centroid[1]),
            spatial_group=str(record.metadata.get("spatial_group", "unknown")),
            nearest_index_m=float(nearest_any[position]),
            nearest_same_label_index_m=float(nearest_same[position]),
        )
    return located


def latitude_band(latitude: float, *, degrees: float) -> str:
    """Return the fixed band a latitude falls in, as a stable sortable label."""

    if not math.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise ValueError(f"latitude must be a finite degree value: {latitude}")
    if not math.isfinite(degrees) or degrees <= 0:
        raise ValueError("degrees must be a positive band width")
    lower = math.floor(latitude / degrees) * degrees
    return f"{lower:+07.2f}..{lower + degrees:+07.2f}"


def latitude_slices(
    evaluations: Sequence[QueryEvaluation],
    geography: Mapping[str, QueryGeography],
    *,
    degrees: float,
) -> dict[str, Any]:
    """Group per-query results into fixed latitude bands."""

    grouped: dict[str, list[QueryEvaluation]] = {}
    for evaluation in evaluations:
        located = geography.get(evaluation.query_id)
        if located is None:
            continue
        band = latitude_band(located.latitude, degrees=degrees)
        grouped.setdefault(band, []).append(evaluation)

    return {
        band: {
            **summarize_queries(members).to_dict(),
            "below_minimum_queries": len(members) < MINIMUM_SLICE_QUERIES,
        }
        for band, members in sorted(grouped.items())
    }


def spatial_group_spread(
    evaluations: Sequence[QueryEvaluation],
    geography: Mapping[str, QueryGeography],
) -> dict[str, Any]:
    """Describe how much quality varies across cells, without per-cell claims.

    The split consumed 725 cells but only a few carry queries, most of them a
    handful each. A per-cell mAP@k would therefore be mostly sampling noise. The
    spread across cells is still meaningful: it separates a representation that
    works evenly everywhere from one that collapses in particular places.
    """

    per_group: dict[str, list[float]] = {}
    for evaluation in evaluations:
        located = geography.get(evaluation.query_id)
        if located is None:
            continue
        per_group.setdefault(located.spatial_group, []).append(
            evaluation.average_precision_at_k
        )
    if not per_group:
        return {"groups": 0}

    means = np.asarray([float(np.mean(values)) for values in per_group.values()])
    sizes = np.asarray([len(values) for values in per_group.values()])
    return {
        "groups": int(means.size),
        "queries_per_group": {
            "minimum": int(sizes.min()),
            "median": float(np.median(sizes)),
            "maximum": int(sizes.max()),
        },
        "group_mean_map_at_k": {
            "minimum": float(means.min()),
            "p10": float(np.percentile(means, 10)),
            "median": float(np.median(means)),
            "p90": float(np.percentile(means, 90)),
            "maximum": float(means.max()),
            "standard_deviation": float(means.std(ddof=0)),
        },
        "interpretation": (
            "spread across cells, not per-cell quality; most cells hold too few "
            "queries to support an individual score"
        ),
    }


def _top_one_correct(evaluation: QueryEvaluation) -> bool:
    return bool(evaluation.relevance) and evaluation.relevance[0] == 1


def agreement(
    per_store: Mapping[str, Sequence[QueryEvaluation]], *, k: int
) -> dict[str, Any]:
    """Measure whether representations differ broadly or on a few queries.

    Two representations can post similar aggregates while ranking almost
    disjoint results. Overlap@k measures that directly, and the correctness
    census says whether one representation's advantage is spread across the
    corpus or concentrated in queries the others simply lose.
    """

    names = sorted(per_store)
    if len(names) < 2:
        return {"stores": names, "note": "agreement needs at least two stores"}

    indexed = {
        name: {evaluation.query_id: evaluation for evaluation in per_store[name]}
        for name in names
    }
    shared = sorted(set.intersection(*(set(values) for values in indexed.values())))
    if not shared:
        return {"stores": names, "note": "stores share no evaluated queries"}

    pairwise = {}
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            overlaps = [
                len(
                    set(indexed[left][query_id].ranked_ids)
                    & set(indexed[right][query_id].ranked_ids)
                )
                / k
                for query_id in shared
            ]
            pairwise[f"{left} vs {right}"] = {
                "mean_overlap_at_k": float(np.mean(overlaps)),
                "queries": len(shared),
            }

    census: dict[str, int] = {}
    correct_counts = []
    for query_id in shared:
        correct = tuple(name for name in names if _top_one_correct(indexed[name][query_id]))
        correct_counts.append(len(correct))
        key = ", ".join(correct) if correct else "none"
        census[key] = census.get(key, 0) + 1

    counts = np.asarray(correct_counts)
    return {
        "stores": names,
        "queries": len(shared),
        "pairwise_overlap": dict(sorted(pairwise.items())),
        "top_one_correctness": {
            "all_stores_correct": int((counts == len(names)).sum()),
            "no_store_correct": int((counts == 0).sum()),
            "exactly_one_store_correct": int((counts == 1).sum()),
            "disputed": int(((counts > 0) & (counts < len(names))).sum()),
        },
        "correct_store_census": dict(
            sorted(census.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def failure_taxonomy(
    per_store: Mapping[str, Sequence[QueryEvaluation]],
    labels: Mapping[str, str],
) -> dict[str, Any]:
    """Record what comes back when a ranking is wrong.

    A wrong neighbour is not a uniform event. Which class arrives instead says
    whether a representation confuses two genuinely similar surfaces or fails
    unsystematically, and the queries every representation misses mark the
    corpus's own hard cases rather than any one model's weakness.
    """

    names = sorted(per_store)
    indexed = {
        name: {evaluation.query_id: evaluation for evaluation in per_store[name]}
        for name in names
    }
    shared = (
        sorted(set.intersection(*(set(values) for values in indexed.values())))
        if names
        else []
    )

    universally_hard = [
        query_id
        for query_id in shared
        if not any(_top_one_correct(indexed[name][query_id]) for name in names)
    ]

    confusion: dict[str, dict[str, dict[str, int]]] = {}
    for name in names:
        per_label: dict[str, dict[str, int]] = {}
        for evaluation in per_store[name]:
            for item_id, relevant in zip(
                evaluation.ranked_ids, evaluation.relevance, strict=False
            ):
                if relevant:
                    continue
                retrieved_label = labels.get(item_id, "unknown")
                bucket = per_label.setdefault(evaluation.label, {})
                bucket[retrieved_label] = bucket.get(retrieved_label, 0) + 1
        confusion[name] = {
            label: dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
            for label, counts in sorted(per_label.items())
        }

    return {
        "universally_hard_queries": {
            "count": len(universally_hard),
            "share_of_queries": float(len(universally_hard) / len(shared)) if shared else 0.0,
            "examples": [
                {"query_id": query_id, "label": labels.get(query_id, "unknown")}
                for query_id in universally_hard[:MAXIMUM_LISTED_QUERIES]
            ],
        },
        "retrieved_label_when_wrong": confusion,
    }


def distance_slices(
    evaluations: Sequence[QueryEvaluation],
    geography: Mapping[str, QueryGeography],
    *,
    attribute: str,
) -> dict[str, Any]:
    """Group per-query results into quartiles of a geographic distance.

    Quartiles rather than a correlation coefficient: the relationship need not be
    linear or monotonic, and four labelled buckets with their metre ranges say
    what a single number would hide.
    """

    paired = [
        (float(getattr(geography[evaluation.query_id], attribute)), evaluation)
        for evaluation in evaluations
        if evaluation.query_id in geography
        and math.isfinite(float(getattr(geography[evaluation.query_id], attribute)))
    ]
    if len(paired) < 4:
        return {"note": "too few located queries to form quartiles"}

    distances = np.asarray([value for value, _ in paired])
    edges = [float(np.quantile(distances, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
    # Bucket by rank so equal distances cannot pile every query into one bin.
    order = np.argsort(distances, kind="stable")
    buckets: list[list[QueryEvaluation]] = [[], [], [], []]
    for rank, position in enumerate(order):
        buckets[min(rank * 4 // len(order), 3)].append(paired[position][1])

    quartiles = {}
    for number, members in enumerate(buckets, start=1):
        if not members:
            continue
        values = [
            float(getattr(geography[evaluation.query_id], attribute)) for evaluation in members
        ]
        quartiles[f"q{number}"] = {
            **summarize_queries(members).to_dict(),
            "distance_m": {"minimum": min(values), "maximum": max(values)},
        }
    return {"attribute": attribute, "quartile_edges_m": edges, "quartiles": quartiles}


def analyze(
    per_store: Mapping[str, Sequence[QueryEvaluation]],
    records: Sequence[ImageRecord],
    *,
    k: int,
    latitude_band_degrees: float = 5.0,
) -> dict[str, Any]:
    """Produce every slice for one already-evaluated set of stores."""

    if not per_store:
        raise ValueError("at least one evaluated store is required")
    geography = query_geography(records)
    labels = {
        record.item_id: record.label for record in records if record.label is not None
    }

    return {
        "k": k,
        "latitude_band_degrees": latitude_band_degrees,
        "aggregate": {
            name: summarize_queries(evaluations).to_dict()
            for name, evaluations in sorted(per_store.items())
        },
        "geographic": {
            "latitude_bands": {
                name: latitude_slices(
                    evaluations, geography, degrees=latitude_band_degrees
                )
                for name, evaluations in sorted(per_store.items())
            },
            "spatial_group_spread": {
                name: spatial_group_spread(evaluations, geography)
                for name, evaluations in sorted(per_store.items())
            },
        },
        "agreement": agreement(per_store, k=k),
        "failures": failure_taxonomy(per_store, labels),
        "distance": {
            attribute: {
                name: distance_slices(evaluations, geography, attribute=attribute)
                for name, evaluations in sorted(per_store.items())
            }
            for attribute in ("nearest_index_m", "nearest_same_label_index_m")
        },
    }
