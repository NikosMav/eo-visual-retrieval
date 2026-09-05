"""Slices of published per-query results must not invent or lose information."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pytest

from eo_visual_retrieval.analysis import (
    agreement,
    analyze,
    distance_slices,
    failure_taxonomy,
    latitude_band,
    latitude_slices,
    query_geography,
    spatial_group_spread,
)
from eo_visual_retrieval.evaluation import QueryEvaluation, summarize_queries
from eo_visual_retrieval.models import ImageRecord


def _record(
    item_id: str,
    label: str,
    split: Literal["index", "query"],
    lonlat: tuple[float, float],
    group: str = "cell:0:0",
) -> ImageRecord:
    return ImageRecord(
        item_id=item_id,
        path=item_id,
        split=split,
        source="eurosat-ms-v1",
        metadata={"centroid_lonlat": [lonlat[0], lonlat[1]], "spatial_group": group},
        label=label,
    )


def _evaluation(
    query_id: str,
    label: str,
    *,
    ranked: tuple[str, ...],
    relevance: tuple[int, ...],
    score: float = 0.5,
) -> QueryEvaluation:
    return QueryEvaluation(
        query_id=query_id,
        label=label,
        precision_at_k=score,
        recall_at_k=score,
        average_precision_at_k=score,
        ndcg_at_k=score,
        ranked_ids=ranked,
        ranked_scores=tuple(1.0 for _ in ranked),
        relevance=relevance,
    )


def _corpus() -> list[ImageRecord]:
    records = [
        _record("q_south", "Forest", "query", (10.0, 36.0), "cell:a"),
        _record("q_mid", "Forest", "query", (10.0, 47.0), "cell:b"),
        _record("q_north", "River", "query", (10.0, 61.0), "cell:c"),
        _record("q_north2", "River", "query", (10.1, 61.1), "cell:c"),
    ]
    records += [
        _record("i_south", "Forest", "index", (10.0, 36.2), "cell:a"),
        _record("i_mid", "Forest", "index", (10.0, 47.9), "cell:b"),
        _record("i_north", "River", "index", (10.0, 61.4), "cell:c"),
    ]
    return records


def test_latitude_bands_are_fixed_and_sortable() -> None:
    assert latitude_band(47.02, degrees=5.0) == "+045.00..+050.00"
    assert latitude_band(34.72, degrees=5.0) == "+030.00..+035.00"
    # Sorting the labels must order the bands south to north.
    labels = [latitude_band(value, degrees=5.0) for value in (61.0, 36.0, 47.0)]
    assert sorted(labels) == [
        "+035.00..+040.00",
        "+045.00..+050.00",
        "+060.00..+065.00",
    ]

    with pytest.raises(ValueError, match="finite degree value"):
        latitude_band(float("nan"), degrees=5.0)
    with pytest.raises(ValueError, match="positive band width"):
        latitude_band(10.0, degrees=0.0)


def test_slices_pool_back_to_the_published_aggregate() -> None:
    """A slice that does not reconcile with the headline is a broken slice.

    Latitude bands partition the queries, so the query-weighted mean over bands
    must equal the aggregate the evaluator publishes. This is the check that
    catches a query silently dropped or double-counted by a grouping.
    """

    records = _corpus()
    geography = query_geography(records)
    evaluations = [
        _evaluation("q_south", "Forest", ranked=("i_south",), relevance=(1,), score=0.9),
        _evaluation("q_mid", "Forest", ranked=("i_mid",), relevance=(1,), score=0.5),
        _evaluation("q_north", "River", ranked=("i_north",), relevance=(1,), score=0.2),
        _evaluation("q_north2", "River", ranked=("i_south",), relevance=(0,), score=0.0),
    ]

    bands = latitude_slices(evaluations, geography, degrees=5.0)
    published = summarize_queries(evaluations).to_dict()

    assert sum(band["evaluated_queries"] for band in bands.values()) == len(evaluations)
    pooled = sum(
        band["map_at_k"] * band["evaluated_queries"] for band in bands.values()
    ) / len(evaluations)
    assert pooled == pytest.approx(published["map_at_k"])


def test_thin_slices_are_flagged_rather_than_dropped() -> None:
    records = _corpus()
    geography = query_geography(records)
    evaluations = [
        _evaluation("q_south", "Forest", ranked=("i_south",), relevance=(1,)),
    ]
    bands = latitude_slices(evaluations, geography, degrees=5.0)

    assert bands["+035.00..+040.00"]["below_minimum_queries"] is True


def test_group_spread_reports_distribution_not_per_cell_scores() -> None:
    records = _corpus()
    geography = query_geography(records)
    evaluations = [
        _evaluation("q_south", "Forest", ranked=("i_south",), relevance=(1,), score=1.0),
        _evaluation("q_mid", "Forest", ranked=("i_mid",), relevance=(1,), score=0.0),
        _evaluation("q_north", "River", ranked=("i_north",), relevance=(1,), score=0.5),
        _evaluation("q_north2", "River", ranked=("i_north",), relevance=(1,), score=0.5),
    ]
    spread = spatial_group_spread(evaluations, geography)

    assert spread["groups"] == 3
    assert spread["group_mean_map_at_k"]["minimum"] == pytest.approx(0.0)
    assert spread["group_mean_map_at_k"]["maximum"] == pytest.approx(1.0)
    assert spread["queries_per_group"]["maximum"] == 2
    # The report must not expose a score attributed to one named cell.
    assert "cell:a" not in str(spread)


def test_agreement_separates_broad_gain_from_a_disputed_few() -> None:
    per_store = {
        "alpha": [
            _evaluation("a", "Forest", ranked=("x", "y"), relevance=(1, 0)),
            _evaluation("b", "River", ranked=("x", "y"), relevance=(0, 0)),
        ],
        "beta": [
            _evaluation("a", "Forest", ranked=("x", "z"), relevance=(1, 0)),
            _evaluation("b", "River", ranked=("w", "v"), relevance=(1, 0)),
        ],
    }
    result = agreement(per_store, k=2)

    assert result["queries"] == 2
    assert result["pairwise_overlap"]["alpha vs beta"]["mean_overlap_at_k"] == pytest.approx(
        (1 / 2 + 0 / 2) / 2
    )
    assert result["top_one_correctness"] == {
        "all_stores_correct": 1,
        "no_store_correct": 0,
        "exactly_one_store_correct": 1,
        "disputed": 1,
    }
    assert result["correct_store_census"]["beta"] == 1


def test_agreement_needs_two_stores() -> None:
    only = {"alpha": [_evaluation("a", "Forest", ranked=("x",), relevance=(1,))]}

    assert "note" in agreement(only, k=1)


def test_failure_taxonomy_names_what_came_back_instead() -> None:
    labels = {"x": "River", "y": "Highway", "q": "Forest"}
    per_store = {
        "alpha": [_evaluation("q", "Forest", ranked=("x", "y"), relevance=(0, 0))],
        "beta": [_evaluation("q", "Forest", ranked=("x", "x"), relevance=(0, 0))],
    }
    result = failure_taxonomy(per_store, labels)

    assert result["universally_hard_queries"]["count"] == 1
    assert result["universally_hard_queries"]["share_of_queries"] == pytest.approx(1.0)
    assert result["retrieved_label_when_wrong"]["alpha"]["Forest"] == {
        "Highway": 1,
        "River": 1,
    }
    assert result["retrieved_label_when_wrong"]["beta"]["Forest"] == {"River": 2}


def test_distance_quartiles_cover_every_located_query() -> None:
    records = _corpus()
    geography = query_geography(records)
    evaluations = [
        _evaluation(item_id, "Forest", ranked=("i_south",), relevance=(1,), score=value)
        for item_id, value in zip(
            ("q_south", "q_mid", "q_north", "q_north2"), (0.1, 0.4, 0.7, 1.0), strict=True
        )
    ]
    result = distance_slices(evaluations, geography, attribute="nearest_index_m")

    assert result["attribute"] == "nearest_index_m"
    assert len(result["quartile_edges_m"]) == 5
    covered = sum(bucket["evaluated_queries"] for bucket in result["quartiles"].values())
    assert covered == len(evaluations)


def test_geography_measures_both_any_and_same_label_neighbours() -> None:
    located = query_geography(_corpus())

    north = located["q_north2"]
    assert north.label == "River"
    # Its nearest index item of any label and of its own label are both real
    # distances in metres, and the same-label one cannot be nearer.
    assert 0 < north.nearest_index_m < 1e6
    assert north.nearest_same_label_index_m >= north.nearest_index_m
    assert math.isfinite(north.nearest_same_label_index_m)


def test_geography_requires_both_partitions() -> None:
    with pytest.raises(ValueError, match="both query and index items"):
        query_geography([_record("q", "Forest", "query", (10.0, 40.0))])


def test_analyze_produces_every_section() -> None:
    records = _corpus()
    per_store = {
        "alpha": [
            _evaluation("q_south", "Forest", ranked=("i_south",), relevance=(1,)),
            _evaluation("q_mid", "Forest", ranked=("i_mid",), relevance=(1,)),
            _evaluation("q_north", "River", ranked=("i_north",), relevance=(1,)),
            _evaluation("q_north2", "River", ranked=("i_south",), relevance=(0,)),
        ],
        "beta": [
            _evaluation("q_south", "Forest", ranked=("i_mid",), relevance=(0,)),
            _evaluation("q_mid", "Forest", ranked=("i_mid",), relevance=(1,)),
            _evaluation("q_north", "River", ranked=("i_north",), relevance=(1,)),
            _evaluation("q_north2", "River", ranked=("i_north",), relevance=(1,)),
        ],
    }
    result = analyze(per_store, records, k=1)

    assert set(result) == {
        "k",
        "latitude_band_degrees",
        "aggregate",
        "geographic",
        "agreement",
        "failures",
        "distance",
    }
    assert set(result["distance"]) == {"nearest_index_m", "nearest_same_label_index_m"}
    assert result["aggregate"]["alpha"]["evaluated_queries"] == 4
    assert np.isclose(result["latitude_band_degrees"], 5.0)


def test_analyze_rejects_an_empty_request() -> None:
    with pytest.raises(ValueError, match="at least one evaluated store"):
        analyze({}, _corpus(), k=1)
