"""A fitted PCA projection that outlives the run that produced it.

PCA is the one representation this project fits itself, so its learned basis is
the one thing that cannot be recovered from a public checkpoint. Without a saved
basis an image outside the original manifest can never be placed in the same
space, which blocks any query-by-upload surface. Persisting the projection keeps
the embedding of a new image identical to the embedding it would have received
during the run that fitted the basis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eo_visual_retrieval.vectors import l2_normalize


@dataclass(frozen=True)
class PcaProjection:
    """Mean and components of a PCA basis fitted on the index partition."""

    mean: NDArray[np.float32]
    components: NDArray[np.float32]
    image_size: int
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.components.ndim != 2:
            raise ValueError("components must be a two-dimensional matrix")
        if self.mean.ndim != 1 or self.mean.shape[0] != self.components.shape[1]:
            raise ValueError("mean must have one value per input feature")
        if self.image_size < 8:
            raise ValueError("image_size must be at least 8")
        expected_features = self.image_size * self.image_size * 3
        if self.components.shape[1] != expected_features:
            raise ValueError(
                f"components expect {self.components.shape[1]} features, but "
                f"image_size {self.image_size} produces {expected_features}"
            )

    @property
    def dimension(self) -> int:
        return int(self.components.shape[0])

    def transform(self, pixels: NDArray[np.float32]) -> NDArray[np.float32]:
        """Project flattened 0-1 RGB pixels and L2-normalize, as scikit-learn does."""

        matrix = np.asarray(pixels, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.components.shape[1]:
            raise ValueError(
                f"pixels must be a two-dimensional matrix with "
                f"{self.components.shape[1]} columns"
            )
        projected = (matrix - self.mean) @ self.components.T
        return l2_normalize(np.asarray(projected, dtype=np.float32))

    def embed_images(self, paths: list[Path]) -> NDArray[np.float32]:
        """Embed local RGB images with the preprocessing this basis was fitted on."""

        from eo_visual_retrieval.embeddings.pca import load_flat_rgb

        if not paths:
            raise ValueError("at least one image is required")
        return self.transform(load_flat_rgb(paths, image_size=self.image_size))

    def save(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary,
            mean=np.asarray(self.mean, dtype=np.float32),
            components=np.asarray(self.components, dtype=np.float32),
            image_size=np.asarray(self.image_size, dtype=np.int64),
            seed=np.asarray(self.seed, dtype=np.int64),
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True), dtype=np.str_),
        )
        temporary.replace(output)

    @classmethod
    def load(cls, path: Path) -> PcaProjection:
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                mean=np.asarray(archive["mean"], dtype=np.float32),
                components=np.asarray(archive["components"], dtype=np.float32),
                image_size=int(archive["image_size"].item()),
                seed=int(archive["seed"].item()),
                metadata=json.loads(str(archive["metadata"].item())),
            )
