from pathlib import Path

import numpy as np

from eo_visual_retrieval.embeddings.store import EmbeddingStore


def test_embedding_store_round_trip(tmp_path: Path) -> None:
    expected = EmbeddingStore(
        ids=("a", "b"),
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        labels=("x", None),
        splits=("index", "query"),
        metadata={"backend": "test"},
    )
    path = tmp_path / "embeddings.npz"
    expected.save(path)
    actual = EmbeddingStore.load(path)
    assert actual.ids == expected.ids
    assert actual.labels == expected.labels
    assert actual.splits == expected.splits
    assert actual.metadata == expected.metadata
    np.testing.assert_array_equal(actual.vectors, expected.vectors)
