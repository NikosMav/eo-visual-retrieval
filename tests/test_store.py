from pathlib import Path

import numpy as np
import pytest

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


def test_unlabeled_and_empty_labels_survive_a_round_trip(tmp_path: Path) -> None:
    """An unlabeled row is skipped by the evaluator; "" is an ordinary class."""

    expected = EmbeddingStore(
        ids=("a", "b", "c"),
        vectors=np.eye(3, 2, dtype=np.float32) + 0.5,
        labels=(None, "", "forest"),
        splits=("index", "index", "query"),
    )
    path = tmp_path / "labels.npz"
    expected.save(path)

    assert EmbeddingStore.load(path).labels == (None, "", "forest")


def test_stores_written_before_label_presence_still_load(tmp_path: Path) -> None:
    """Existing local artifacts must keep loading after the format gained a field."""

    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        ids=np.asarray(["a", "b"], dtype=np.str_),
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        labels=np.asarray(["forest", ""], dtype=np.str_),
        splits=np.asarray(["index", "query"], dtype=np.str_),
        metadata=np.asarray('{"backend": "test"}', dtype=np.str_),
    )

    loaded = EmbeddingStore.load(path)

    assert loaded.labels == ("forest", None)
    assert loaded.metadata == {"backend": "test"}


def test_mismatched_label_presence_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.npz"
    np.savez_compressed(
        path,
        ids=np.asarray(["a", "b"], dtype=np.str_),
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        labels=np.asarray(["forest", "water"], dtype=np.str_),
        label_present=np.asarray([True], dtype=np.bool_),
        splits=np.asarray(["index", "query"], dtype=np.str_),
        metadata=np.asarray("{}", dtype=np.str_),
    )

    with pytest.raises(ValueError, match="label presence does not match"):
        EmbeddingStore.load(path)
