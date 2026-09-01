import numpy as np

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
