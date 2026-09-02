"""Backend dispatch when embedding an image that is not already in a store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eo_visual_retrieval.embeddings.encode import embed_query_image
from eo_visual_retrieval.embeddings.projection import PcaProjection
from eo_visual_retrieval.embeddings.store import EmbeddingStore

IMAGE_SIZE = 8
FEATURES = IMAGE_SIZE * IMAGE_SIZE * 3


def _image(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "query.png"
    Image.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) + 30).save(path)
    return path


def _projection(dimension: int = 2, image_size: int = IMAGE_SIZE) -> PcaProjection:
    features = image_size * image_size * 3
    return PcaProjection(
        mean=np.zeros(features, dtype=np.float32),
        components=np.eye(dimension, features, dtype=np.float32) + 0.5,
        image_size=image_size,
        seed=42,
    )


def _store(backend: str, dimension: int = 2, **metadata: Any) -> EmbeddingStore:
    return EmbeddingStore(
        ids=("a", "b"),
        vectors=np.eye(2, dimension, dtype=np.float32),
        labels=("x", "x"),
        splits=("index", "query"),
        metadata={"backend": backend, **metadata},
    )


def test_pca_query_image_is_embedded_into_the_stored_space(tmp_path: Path) -> None:
    vector = embed_query_image(
        _image(tmp_path),
        store=_store("pca", image_size=IMAGE_SIZE),
        projection=_projection(),
    )

    assert vector.shape == (2,)
    assert vector.dtype == np.float32
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)


def test_pca_query_requires_the_fitted_projection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs the fitted projection"):
        embed_query_image(_image(tmp_path), store=_store("pca"), projection=None)


def test_projection_must_match_the_stores_preprocessing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match the store's recorded image_size"):
        embed_query_image(
            _image(tmp_path),
            store=_store("pca", image_size=16),
            projection=_projection(image_size=IMAGE_SIZE),
        )


def test_multispectral_backends_refuse_an_rgb_upload(tmp_path: Path) -> None:
    """Approximating a 13-band input from RGB would silently invalidate the ranking."""

    for backend in ("ssl4eo-s12", "terramind"):
        with pytest.raises(ValueError, match="13-band archive members"):
            embed_query_image(_image(tmp_path), store=_store(backend))


def test_unknown_backend_is_reported_rather_than_guessed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not supported for backend 'custom'"):
        embed_query_image(_image(tmp_path), store=_store("custom"))


def test_missing_image_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="query image does not exist"):
        embed_query_image(tmp_path / "absent.png", store=_store("pca"))


def test_dimension_mismatch_between_projection_and_store_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dimensions but the store holds"):
        embed_query_image(
            _image(tmp_path),
            store=_store("pca", dimension=3, image_size=IMAGE_SIZE),
            projection=_projection(dimension=2),
        )
