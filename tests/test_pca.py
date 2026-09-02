"""PCA fitting boundary, normalization, and reuse of a persisted basis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eo_visual_retrieval.embeddings.pca import fit_pca_projection, load_flat_rgb, pca_embeddings
from eo_visual_retrieval.embeddings.projection import PcaProjection

pytest.importorskip("sklearn")

IMAGE_SIZE = 8
FEATURES = IMAGE_SIZE * IMAGE_SIZE * 3


def _write_images(root: Path, count: int) -> list[Path]:
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    generator = np.random.default_rng(7)
    for position in range(count):
        pixels = generator.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        path = root / f"image-{position:02d}.png"
        Image.fromarray(pixels).save(path)
        paths.append(path)
    return paths


def test_pca_is_fitted_on_index_rows_only(tmp_path: Path) -> None:
    """A query image must not influence the basis it is later projected into."""

    paths = _write_images(tmp_path, 6)
    splits = ["index"] * 4 + ["query"] * 2
    pixels = load_flat_rgb(paths, image_size=IMAGE_SIZE)

    projection = fit_pca_projection(pixels, splits, components=3, image_size=IMAGE_SIZE)
    index_only = fit_pca_projection(
        pixels[:4], ["index"] * 4, components=3, image_size=IMAGE_SIZE
    )

    np.testing.assert_allclose(projection.mean, index_only.mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(projection.components, index_only.components, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(projection.mean, pixels[:4].mean(axis=0), rtol=1e-5, atol=1e-6)
    assert projection.metadata["fit_partition"] == "index"
    assert projection.metadata["fit_items"] == 4


def test_pca_embeddings_are_normalized_and_reproduced_by_the_saved_basis(tmp_path: Path) -> None:
    paths = _write_images(tmp_path, 6)
    splits = ["index"] * 4 + ["query"] * 2

    vectors, projection = pca_embeddings(paths, splits, components=3, image_size=IMAGE_SIZE)

    assert vectors.shape == (6, 3)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6, atol=1e-6)

    saved = tmp_path / "projection.npz"
    projection.save(saved)
    reloaded = PcaProjection.load(saved)

    np.testing.assert_allclose(reloaded.embed_images(paths), vectors, rtol=1e-5, atol=1e-6)
    assert reloaded.image_size == IMAGE_SIZE
    assert reloaded.seed == projection.seed
    assert reloaded.dimension == 3


def test_pca_rejects_more_components_than_index_items(tmp_path: Path) -> None:
    paths = _write_images(tmp_path, 4)
    splits = ["index"] * 2 + ["query"] * 2

    with pytest.raises(ValueError, match="components must be between 1 and 2"):
        pca_embeddings(paths, splits, components=3, image_size=IMAGE_SIZE)


def test_projection_rejects_inconsistent_geometry() -> None:
    components = np.zeros((2, FEATURES), dtype=np.float32)

    with pytest.raises(ValueError, match="one value per input feature"):
        PcaProjection(
            mean=np.zeros(FEATURES - 1, dtype=np.float32),
            components=components,
            image_size=IMAGE_SIZE,
            seed=1,
        )
    with pytest.raises(ValueError, match="image_size 16 produces"):
        PcaProjection(
            mean=np.zeros(FEATURES, dtype=np.float32),
            components=components,
            image_size=16,
            seed=1,
        )


def test_projection_rejects_wrong_input_width() -> None:
    projection = PcaProjection(
        mean=np.zeros(FEATURES, dtype=np.float32),
        components=np.eye(2, FEATURES, dtype=np.float32),
        image_size=IMAGE_SIZE,
        seed=1,
    )

    with pytest.raises(ValueError, match=f"{FEATURES} columns"):
        projection.transform(np.ones((1, FEATURES - 1), dtype=np.float32))
