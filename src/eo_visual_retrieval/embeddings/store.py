"""Portable storage for embeddings and retrieval metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EmbeddingStore:
    ids: tuple[str, ...]
    vectors: NDArray[np.float32]
    labels: tuple[str | None, ...]
    splits: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = len(self.ids)
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a two-dimensional matrix")
        if self.vectors.shape[0] != count:
            raise ValueError("ids and vectors must contain the same number of items")
        if len(self.labels) != count or len(self.splits) != count:
            raise ValueError("ids, labels, and splits must have matching lengths")
        if len(set(self.ids)) != count:
            raise ValueError("embedding IDs must be unique")
        if not set(self.splits).issubset({"index", "query"}):
            raise ValueError("embedding splits must be 'index' or 'query'")

    def save(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary,
            ids=np.asarray(self.ids, dtype=np.str_),
            vectors=np.asarray(self.vectors, dtype=np.float32),
            labels=np.asarray([label or "" for label in self.labels], dtype=np.str_),
            splits=np.asarray(self.splits, dtype=np.str_),
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True), dtype=np.str_),
        )
        temporary.replace(output)

    @classmethod
    def load(cls, path: Path) -> EmbeddingStore:
        with np.load(path, allow_pickle=False) as archive:
            labels = tuple(str(value) or None for value in archive["labels"].tolist())
            metadata = json.loads(str(archive["metadata"].item()))
            return cls(
                ids=tuple(str(value) for value in archive["ids"].tolist()),
                vectors=np.asarray(archive["vectors"], dtype=np.float32),
                labels=labels,
                splits=tuple(str(value) for value in archive["splits"].tolist()),
                metadata=metadata,
            )
