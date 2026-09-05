"""Exact text/image retrieval, metadata prefilters, and explainable score fusion."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PureWindowsPath
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from eo_visual_retrieval.embeddings.remoteclip import MultimodalEncoder
from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.hashing import file_sha256
from eo_visual_retrieval.manifests import read_jsonl
from eo_visual_retrieval.models import ImageRecord
from eo_visual_retrieval.search_plan import SearchPlan, filter_checks, scene_metadata
from eo_visual_retrieval.vectors import l2_normalize


class MultimodalSearch:
    def __init__(
        self,
        records: Sequence[ImageRecord],
        store: EmbeddingStore,
        encoder: MultimodalEncoder,
        *,
        image_root: Path,
    ) -> None:
        if (
            tuple(r.item_id for r in records) != store.ids
            or tuple(r.split for r in records) != store.splits
            or tuple(r.label for r in records) != store.labels
        ):
            raise ValueError("manifest IDs, order, splits and labels must match the store")
        if any(store.metadata.get(key) != value for key, value in encoder.metadata.items()):
            raise ValueError(
                "encoder and store embedding spaces differ; rebuild with embed-remoteclip"
            )
        if store.vectors.shape[1] != encoder.metadata["dimension"]:
            raise ValueError("encoder dimension does not match the store")
        self.store = store
        self.encoder = encoder
        self._records = {record.item_id: record for record in records}
        self._root = image_root.resolve()
        self._positions = {item_id: i for i, item_id in enumerate(store.ids)}
        self._index = [i for i, split in enumerate(store.splits) if split == "index"]
        if not self._index:
            raise ValueError("store contains no index items")
        self._vectors = l2_normalize(store.vectors)
        self._metadata = [scene_metadata(record) for record in records]
        self._inference_lock = Lock()
        for record in records:
            path = self.image_path(record.item_id)
            if not path.is_file():
                raise ValueError(f"missing corpus image: {record.item_id}")
            expected = record.metadata.get("sha256")
            if expected is not None and file_sha256(path) != expected:
                raise ValueError(f"image checksum disagrees with manifest: {record.item_id}")

    @classmethod
    def load(
        cls,
        *,
        manifest: Path,
        embeddings: Path,
        image_root: Path,
        encoder: MultimodalEncoder,
    ) -> MultimodalSearch:
        store = EmbeddingStore.load(embeddings)
        if store.metadata.get("manifest_sha256") != file_sha256(manifest):
            raise ValueError("manifest SHA-256 does not match the embedding store")
        return cls(read_jsonl(manifest), store, encoder, image_root=image_root)

    def image_path(self, item_id: str) -> Path:
        if item_id not in self._records:
            raise ValueError("unknown corpus item")
        relative = self._records[item_id].path
        windows = PureWindowsPath(relative)
        if (
            Path(relative).is_absolute()
            or windows.is_absolute()
            or windows.drive
            or "\\" in relative
        ):
            raise ValueError("image paths must be portable relative paths")
        path = (self._root / relative).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError("corpus image resolves outside the image root")
        return path

    def describe(self) -> dict[str, Any]:
        rows = [self._metadata[i] for i in self._index]
        dates = sorted(row["date"] for row in rows if row["date"] is not None)
        return {
            "index_items": len(rows),
            "model": self.store.metadata["model"],
            "provenance": {**self.encoder.metadata,
                           "manifest_sha256": self.store.metadata.get("manifest_sha256")},
            "query_items": sum(split == "query" for split in self.store.splits),
            "date_range": [dates[0], dates[-1]] if dates else None,
            "metadata_coverage": {
                key: sum(row[key] is not None for row in rows)
                for key in ("date", "centroid_lonlat", "cloud_cover", "collection")
            },
            "examples": [
                {"item_id": item_id, "label": self._records[item_id].label}
                for item_id in self.store.ids
            ],
        }

    def search(
        self,
        plan: SearchPlan,
        *,
        image: Image.Image | None = None,
        item_id: str | None = None,
        text_weight: float = 0.5,
        k: int = 12,
    ) -> dict[str, Any]:
        started = perf_counter()
        if not 1 <= k <= 100:
            raise ValueError("k must be between 1 and 100")
        if not np.isfinite(text_weight) or not 0 <= text_weight <= 1:
            raise ValueError("text weight must be between 0 and 1")
        if image is not None and item_id is not None:
            raise ValueError("choose an uploaded image or a corpus example, not both")
        if item_id is not None and item_id not in self._positions:
            raise ValueError("unknown example item")
        text = plan.text.strip()
        has_image = image is not None or item_id is not None
        if not text and not has_image:
            raise ValueError("provide a description, an image, or both")
        mode = "hybrid" if text and has_image else "text" if text else "image"
        weight = text_weight if mode == "hybrid" else 1.0 if text else 0.0
        pool = [i for i in self._index if self.store.ids[i] != item_id]
        checks = {i: filter_checks(self._metadata[i], plan.filters) for i in pool}
        candidates = [i for i in pool if all(v == "pass" for v in checks[i].values())]
        filter_counts = {
            key: {state: sum(checks[i].get(key) == state for i in pool)
                  for state in ("pass", "fail", "missing")}
            for key, value in plan.filters.to_dict().items() if value is not None
        }
        response: dict[str, Any] = {
            "mode": mode,
            "plan": plan.to_dict(),
            "text_weight": weight,
            "candidate_count": len(candidates),
            "index_count": len(self._index),
            "ranker": "exact-weighted-cosine",
            "model": self.store.metadata["model"],
            "provenance": self.describe()["provenance"],
            "query_input": {"item_id": item_id, "uploaded_image": image is not None,
                            "requested_k": k},
            "diagnostics": {
                "excluded_example": len(self._index) - len(pool),
                "excluded_by_filters": len(pool) - len(candidates),
                "filter_counts": filter_counts,
                "filter_counts_scope": "Each filter independently, after example exclusion; "
                                       "counts overlap. All active filters must pass.",
                "score_meaning": "Cosine similarity, not probability or verified relevance.",
                "timing_scope": "Engine wall time including inference queue; excludes HTTP "
                                "upload/decode, model startup and rendering.",
                "tie_break": "Stable corpus order",
            },
            "results": [],
        }
        if not candidates:
            response["message"] = (
                "No index scenes satisfy all filters. Missing metadata is excluded. "
                "Adjust the displayed filters or prepare a corpus covering this place and period."
            )
            response["diagnostics"]["elapsed_ms"] = (perf_counter() - started) * 1000
            return response
        # One model per process; avoid concurrent model allocations/GPU forwards.
        with self._inference_lock:
            text_vector = self.encoder.encode_text(text) if text else None
            image_vector = (
                self.encoder.encode_image(image)
                if image is not None
                else self._vectors[self._positions[item_id]]
                if item_id
                else None
            )
        matrix = self._vectors[candidates]

        def score(vector: NDArray[np.float32] | None) -> NDArray[np.float32] | None:
            if vector is None:
                return None
            if np.asarray(vector).shape != (matrix.shape[1],):
                raise ValueError("query dimension does not match the index")
            return matrix @ l2_normalize(np.asarray(vector, dtype=np.float32)[None, :])[0]

        text_scores, image_scores = score(text_vector), score(image_vector)
        combined = np.zeros(len(candidates), dtype=np.float32)
        if text_scores is not None:
            combined += weight * text_scores
        if image_scores is not None:
            combined += (1 - weight) * image_scores
        # Stable sort preserves corpus order under ties, including opposing
        # vectors in a 50/50 hybrid. No normalization of a cancelling mixture.
        order = np.argsort(-combined, kind="stable")[:k]

        def ranks(scores: NDArray[np.float32] | None) -> dict[int, int]:
            return {} if scores is None else {
                int(position): rank for rank, position in
                enumerate(np.argsort(-scores, kind="stable"), start=1)
            }

        text_ranks, image_ranks = ranks(text_scores), ranks(image_scores)
        response["results"] = [
            {
                "rank": rank,
                "item_id": self.store.ids[candidates[j]],
                "score": float(combined[j]),
                "text_score": float(text_scores[j]) if text_scores is not None else None,
                "image_score": float(image_scores[j]) if image_scores is not None else None,
                "label": self.store.labels[candidates[j]],
                "metadata": self._metadata[candidates[j]],
                "explanation": {
                    "text_rank": text_ranks.get(int(j)),
                    "image_rank": image_ranks.get(int(j)),
                    "text_contribution": weight * float(text_scores[j])
                    if text_scores is not None else 0.0,
                    "image_contribution": (1 - weight) * float(image_scores[j])
                    if image_scores is not None else 0.0,
                    "filter_checks": checks[candidates[j]],
                    "rank_scope": "All eligible scenes; component ranks use the same pool.",
                },
            }
            for rank, j in enumerate(order, start=1)
        ]
        response["diagnostics"]["elapsed_ms"] = (perf_counter() - started) * 1000
        return response
