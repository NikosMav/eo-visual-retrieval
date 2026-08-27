"""Classical flattened-pixel PCA baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


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


def pca_embeddings(
    paths: list[Path],
    splits: list[str],
    *,
    components: int = 64,
    image_size: int = 64,
    seed: int = 42,
) -> NDArray[np.float32]:
    """Fit PCA on index images only, then transform every image."""

    try:
        from sklearn.decomposition import PCA
    except ImportError as error:
        message = 'PCA support is optional; install with pip install -e ".[ml]"'
        raise RuntimeError(message) from error

    pixels = load_flat_rgb(paths, image_size=image_size)
    index_mask = np.asarray([split == "index" for split in splits])
    index_count = int(index_mask.sum())
    max_components = min(index_count, pixels.shape[1])
    if not 1 <= components <= max_components:
        raise ValueError(f"components must be between 1 and {max_components}")

    model = PCA(n_components=components, random_state=seed)
    model.fit(pixels[index_mask])
    vectors = np.asarray(model.transform(pixels), dtype=np.float32)
    return l2_normalize(vectors)


def l2_normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize a zero-length embedding")
    return np.asarray(vectors / norms, dtype=np.float32)
