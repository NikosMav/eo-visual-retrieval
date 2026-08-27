import numpy as np
import pytest

from eo_visual_retrieval.retrieval import ExactCosineIndex


def test_exact_cosine_returns_ranked_neighbors() -> None:
    index = ExactCosineIndex(
        ["east", "north", "west"],
        np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),
    )
    results = index.search(np.asarray([0.9, 0.1], dtype=np.float32), k=2)
    assert [result.item_id for result in results] == ["east", "north"]
    assert results[0].score > results[1].score


def test_exact_cosine_can_exclude_query_id() -> None:
    index = ExactCosineIndex(
        ["query", "neighbor"],
        np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
    )
    results = index.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        k=2,
        exclude_id="query",
    )
    assert [result.item_id for result in results] == ["neighbor"]


def test_exact_cosine_rejects_zero_query() -> None:
    index = ExactCosineIndex(["a"], np.asarray([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="zero-length"):
        index.search(np.asarray([0.0, 0.0], dtype=np.float32), k=1)
