import numpy as np
import pytest

from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.evaluation import evaluate_store


def test_perfect_label_retrieval_scores_one() -> None:
    store = EmbeddingStore(
        ids=("forest-index", "water-index", "forest-query", "water-query"),
        vectors=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.99, 0.01], [0.01, 0.99]],
            dtype=np.float32,
        ),
        labels=("forest", "water", "forest", "water"),
        splits=("index", "index", "query", "query"),
    )
    summary = evaluate_store(store, k=1)
    assert summary.evaluated_queries == 2
    assert summary.precision_at_k == 1.0
    assert summary.recall_at_k == 1.0
    assert summary.map_at_k == 1.0
    assert summary.ndcg_at_k == 1.0
    assert set(summary.per_class) == {"forest", "water"}
    assert summary.per_class["forest"].evaluated_queries == 1
    assert summary.to_dict()["per_class"]["water"]["precision_at_k"] == 1.0


def test_queries_without_a_usable_label_are_skipped_not_scored() -> None:
    store = EmbeddingStore(
        ids=("forest-index", "forest-query", "unlabeled-query", "orphan-query"),
        vectors=np.asarray(
            [[1.0, 0.0], [0.99, 0.01], [0.5, 0.5], [0.0, 1.0]],
            dtype=np.float32,
        ),
        labels=("forest", "forest", None, "desert"),
        splits=("index", "query", "query", "query"),
    )

    summary = evaluate_store(store, k=1)

    # One query has no label; one has a label no index item shares.
    assert summary.evaluated_queries == 1
    assert summary.skipped_queries == 2
    assert set(summary.per_class) == {"forest"}


def test_metrics_are_scored_against_retrievable_positions_not_the_requested_k() -> None:
    """Asking for more results than the index can return must not deflate scores."""

    store = EmbeddingStore(
        ids=("forest-a", "forest-b", "forest-query"),
        vectors=np.asarray([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]], dtype=np.float32),
        labels=("forest", "forest", "forest"),
        splits=("index", "index", "query"),
    )

    summary = evaluate_store(store, k=2)

    assert summary.evaluated_queries == 1
    assert summary.precision_at_k == 1.0
    assert summary.map_at_k == 1.0
    assert summary.ndcg_at_k == 1.0
    assert summary.recall_at_k == 1.0


def test_partial_relevance_produces_expected_ranked_metrics() -> None:
    store = EmbeddingStore(
        ids=("hit", "miss", "query"),
        vectors=np.asarray([[1.0, 0.0], [0.9, 0.44], [0.99, 0.14]], dtype=np.float32),
        labels=("forest", "water", "forest"),
        splits=("index", "index", "query"),
    )

    summary = evaluate_store(store, k=2)

    # Ranking is [forest, water]: one of two positions is relevant, at rank 1.
    assert summary.precision_at_k == 0.5
    assert summary.recall_at_k == 1.0
    assert summary.map_at_k == 1.0
    assert summary.ndcg_at_k == 1.0


def test_k_above_the_index_size_is_rejected() -> None:
    store = EmbeddingStore(
        ids=("only-index", "query"),
        vectors=np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
        labels=("forest", "forest"),
        splits=("index", "query"),
    )

    with pytest.raises(ValueError, match="k cannot exceed the number of index items"):
        evaluate_store(store, k=2)
