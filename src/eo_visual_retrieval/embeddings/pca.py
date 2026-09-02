"""Classical flattened-pixel PCA baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from eo_visual_retrieval.embeddings.projection import PcaProjection


def load_flat_rgb(paths: list[Path], image_size: int = 64) -> NDArray[np.float32]:
    if image_size < 8:
        raise ValueError("image_size must be at least 8")
    vectors: list[NDArray[np.float32]] = []
    for path in paths:
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((image_size, image_size))
            array = np.asarray(resized, dtype=np.float32) / 255.0
            vectors.append(array.reshape(-1))
    return np.stack(vectors)


def fit_pca_projection(
    pixels: NDArray[np.float32],
    splits: list[str],
    *,
    components: int = 64,
    image_size: int = 64,
    seed: int = 42,
) -> PcaProjection:
    """Fit a reusable PCA basis on index rows only, without class labels."""

    try:
        from sklearn.decomposition import PCA
    except ImportError as error:
        message = 'PCA support is optional; install with pip install -e ".[ml]"'
        raise RuntimeError(message) from error

    if len(splits) != pixels.shape[0]:
        raise ValueError("splits must describe every pixel row")
    index_mask = np.asarray([split == "index" for split in splits])
    index_count = int(index_mask.sum())
    max_components = min(index_count, pixels.shape[1])
    if not 1 <= components <= max_components:
        raise ValueError(f"components must be between 1 and {max_components}")

    model = PCA(n_components=components, random_state=seed)
    model.fit(pixels[index_mask])
    return PcaProjection(
        mean=np.asarray(model.mean_, dtype=np.float32),
        components=np.asarray(model.components_, dtype=np.float32),
        image_size=image_size,
        seed=seed,
        metadata={
            "backend": "pca",
            "components": components,
            "image_size": image_size,
            "seed": seed,
            "fit_partition": "index",
            "fit_items": index_count,
            "preprocessing": "RGB, square resize, 0-1 scaling, flattened pixels",
        },
    )


def pca_embeddings(
    paths: list[Path],
    splits: list[str],
    *,
    components: int = 64,
    image_size: int = 64,
    seed: int = 42,
) -> tuple[NDArray[np.float32], PcaProjection]:
    """Fit PCA on index images only, then transform every image.

    The fitted projection is returned alongside the vectors so it can be saved
    and reused to embed images that were not part of this manifest.
    """

    pixels = load_flat_rgb(paths, image_size=image_size)
    projection = fit_pca_projection(
        pixels,
        splits,
        components=components,
        image_size=image_size,
        seed=seed,
    )
    return projection.transform(pixels), projection
