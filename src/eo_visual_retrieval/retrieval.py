"""Exact cosine retrieval used as the quality reference.

Normalization is delegated to :mod:`eo_visual_retrieval.vectors` so the ranker
and the embedding backends can never disagree about what a unit vector is, and
so both reject the inputs that would otherwise rank silently: a NaN row, whose
norm is not ``0``, and a finite row whose norm overflows float32.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eo_visual_retrieval.vectors import l2_normalize


@dataclass(frozen=True)
class SearchResult:
    item_id: str
    score: float
    position: int


class ExactCosineIndex:
    def __init__(self, ids: list[str], vectors: NDArray[np.float32]) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError("ids and vectors must describe the same two-dimensional matrix")
        if not ids:
            raise ValueError("index must contain at least one vector")
        if len(set(ids)) != len(ids):
            raise ValueError("index IDs must be unique")
        self.ids = tuple(ids)
        self.vectors = l2_normalize(vectors)

    def search(
        self,
        query: NDArray[np.float32],
        *,
        k: int,
        exclude_id: str | None = None,
    ) -> list[SearchResult]:
        if query.ndim != 1 or query.shape[0] != self.vectors.shape[1]:
            raise ValueError("query dimension does not match the index")
        if not 1 <= k <= len(self.ids):
            raise ValueError(f"k must be between 1 and {len(self.ids)}")
        # Normalizing through the shared rule also rejects a NaN query, whose
        # norm is not ``== 0`` and whose NaN scores make ``argsort`` arbitrary.
        unit_query = l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]

        scores = self.vectors @ unit_query
        if exclude_id is not None and exclude_id in self.ids:
            scores = scores.copy()
            scores[self.ids.index(exclude_id)] = -np.inf

        available = len(self.ids) - int(exclude_id in self.ids if exclude_id else False)
        effective_k = min(k, available)
        order = np.argsort(-scores, kind="stable")[:effective_k]
        return [
            SearchResult(
                item_id=self.ids[position],
                score=float(scores[position]),
                position=int(position),
            )
            for position in order
        ]
