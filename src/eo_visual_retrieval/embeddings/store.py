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
            # An unlabeled row and a row labeled "" are different things: the first
            # is skipped by the evaluator, the second is a class of its own. NPZ
            # cannot hold None, so presence travels in its own array.
            label_present=np.asarray(
                [label is not None for label in self.labels], dtype=np.bool_
            ),
            splits=np.asarray(self.splits, dtype=np.str_),
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True), dtype=np.str_),
        )
        temporary.replace(output)

    @classmethod
    def load(cls, path: Path) -> EmbeddingStore:
        with np.load(path, allow_pickle=False) as archive:
            raw_labels = [str(value) for value in archive["labels"].tolist()]
            if "label_present" in archive.files:
                present = [bool(value) for value in archive["label_present"].tolist()]
                if len(present) != len(raw_labels):
                    raise ValueError("embedding store label presence does not match labels")
            else:
                # Stores written before label presence was recorded: an empty
                # string was the only available encoding of "unlabeled".
                present = [bool(label) for label in raw_labels]
            labels = tuple(
                label if is_present else None
                for label, is_present in zip(raw_labels, present, strict=True)
            )
            metadata = json.loads(str(archive["metadata"].item()))
            return cls(
                ids=tuple(str(value) for value in archive["ids"].tolist()),
                vectors=np.asarray(archive["vectors"], dtype=np.float32),
                labels=labels,
                splits=tuple(str(value) for value in archive["splits"].tolist()),
                metadata=metadata,
            )
