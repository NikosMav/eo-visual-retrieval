from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.faiss_benchmark import (
    ann_recall_at_k,
    benchmark_faiss,
    expand_corpus,
    l2_normalize,
)


def test_l2_normalize_returns_unit_float32_rows() -> None:
    actual = l2_normalize(np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))

    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1.0)


def test_expand_corpus_is_deterministic_and_preserves_base_rows() -> None:
    base = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    first = expand_corpus(base, target_size=6, seed=7, noise_std=0.01)
    second = expand_corpus(base, target_size=6, seed=7, noise_std=0.01)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:2], base)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)


def test_ann_recall_is_neighbor_overlap_not_label_recall() -> None:
    exact = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
    approximate = np.asarray([[2, 5], [4, 3]], dtype=np.int64)

    assert ann_recall_at_k(exact, approximate) == pytest.approx(0.75)


def test_benchmark_faiss_reports_exact_and_hnsw_contract() -> None:
    pytest.importorskip("faiss")
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [1.0, 0.05, 0.0],
            [0.0, 1.0, 0.05],
        ],
        dtype=np.float32,
    )
    store = EmbeddingStore(
        ids=("a", "b", "c", "d", "q1", "q2"),
        vectors=vectors,
        labels=("x", "x", "y", "y", "x", "y"),
        splits=("index", "index", "index", "index", "query", "query"),
        metadata={"backend": "test"},
    )

    result = benchmark_faiss(
        store,
        corpus_size=4,
        k=2,
        m=4,
        ef_construction=20,
        ef_search_values=(2, 4),
        threads=1,
        warmups=0,
        repeats=1,
    )

    assert result["exact"]["index"] == "IndexFlatIP"
    assert result["hnsw"]["index"] == "IndexHNSWFlat"
    assert result["workload"]["synthetic_expansion"] is False
    assert [row["ef_search"] for row in result["hnsw"]["searches"]] == [2, 4]
    assert all(0.0 <= row["ann_recall_at_k"] <= 1.0 for row in result["hnsw"]["searches"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_size": 1, "seed": 1, "noise_std": 0.01}, "target_size"),
        ({"target_size": 2, "seed": 1, "noise_std": 0.0}, "noise_std"),
    ],
)
def test_expand_corpus_rejects_invalid_configuration(
    kwargs: dict[str, Any], message: str
) -> None:
    base = np.eye(2, dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        expand_corpus(base, **kwargs)
