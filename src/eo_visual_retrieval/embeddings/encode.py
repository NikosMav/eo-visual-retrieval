"""Embed one new image with the backend that produced an existing store.

Retrieval is only usable when a caller can bring an image the corpus has never
seen. That requires re-running the exact preprocessing and representation the
store was built with, so this module reads the backend configuration recorded in
the store's metadata rather than accepting it as a separate argument that could
silently disagree with the vectors it is compared against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from eo_visual_retrieval.embeddings.projection import PcaProjection
from eo_visual_retrieval.embeddings.store import EmbeddingStore

# Backends that consume an ordinary RGB file. The multispectral encoders read
# 13-band members from a verified source archive, so an RGB upload is not a
# valid input for them and must not be silently approximated.
RGB_QUERY_BACKENDS = ("pca", "dinov2")
ARCHIVE_ONLY_BACKENDS = ("ssl4eo-s12", "terramind")


def embed_query_image(
    image: Path,
    *,
    store: EmbeddingStore,
    projection: PcaProjection | None = None,
    device: str = "auto",
) -> NDArray[np.float32]:
    """Return one unit-length vector comparable with ``store``'s index rows."""

    if not image.is_file():
        raise ValueError(f"query image does not exist: {image}")
    backend = str(store.metadata.get("backend", "unknown"))
    if backend in ARCHIVE_ONLY_BACKENDS:
        raise ValueError(
            f"the {backend} backend embeds 13-band archive members, not RGB files; "
            "query it with --item-id against the prepared benchmark instead"
        )
    if backend not in RGB_QUERY_BACKENDS:
        raise ValueError(
            f"embedding a new image is not supported for backend '{backend}'; "
            f"supported backends are {', '.join(RGB_QUERY_BACKENDS)}"
        )

    if backend == "pca":
        if projection is None:
            raise ValueError(
                "the pca backend needs the fitted projection saved by embed-pca "
                "--projection-output; without it a new image cannot be placed in "
                "the same space"
            )
        recorded_size = store.metadata.get("image_size")
        if recorded_size is not None and int(recorded_size) != projection.image_size:
            raise ValueError(
                f"projection image_size {projection.image_size} does not match the "
                f"store's recorded image_size {int(recorded_size)}"
            )
        vectors = projection.embed_images([image])
    else:
        from eo_visual_retrieval.embeddings.dinov2 import dinov2_embeddings

        model_name = str(store.metadata.get("model", "dinov2_vits14"))
        vectors = dinov2_embeddings([image], model_name=model_name, device=device)

    vector = np.asarray(vectors[0], dtype=np.float32)
    if vector.shape[0] != store.vectors.shape[1]:
        raise ValueError(
            f"embedded query has {vector.shape[0]} dimensions but the store holds "
            f"{store.vectors.shape[1]}"
        )
    return vector
