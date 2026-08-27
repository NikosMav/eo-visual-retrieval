"""Exact cosine retrieval used as the quality reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("index contains a zero-length vector")
        self.ids = tuple(ids)
        self.vectors = np.asarray(vectors / norms, dtype=np.float32)

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
        norm = float(np.linalg.norm(query))
        if norm == 0:
            raise ValueError("query must not be a zero-length vector")

        scores = self.vectors @ (query / norm)
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
