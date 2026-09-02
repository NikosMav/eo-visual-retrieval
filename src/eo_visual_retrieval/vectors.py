"""Vector preparation shared by every embedding backend and search path.

Cosine similarity is implemented as an inner product over unit-length rows, so
normalization is a correctness precondition rather than a convenience. Both the
embedding backends and the Faiss benchmark use this single strict implementation
so an index and its queries can never be prepared by different rules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def l2_normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return a contiguous float32 matrix whose rows have unit L2 norm."""

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("vectors must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("vectors must not contain zero-length rows")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)
