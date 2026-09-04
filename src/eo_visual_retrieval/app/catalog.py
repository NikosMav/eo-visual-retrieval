"""Ranking and provenance for the served comparison surface.

This module deliberately imports nothing web-related, so the logic that decides
what a page claims can be tested without starting a server. It also imports no
model framework: ranking is a matrix-vector product over vectors that were
computed offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from eo_visual_retrieval.embeddings.projection import PcaProjection
from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.hashing import file_sha256
from eo_visual_retrieval.manifests import read_jsonl
from eo_visual_retrieval.models import ImageRecord
from eo_visual_retrieval.retrieval import ExactCosineIndex

PCA_BACKEND = "pca"


@dataclass(frozen=True)
class RankedResult:
    """One retrieved item. ``relevant`` is None when the query carries no label."""

    item_id: str
    score: float
    relevant: bool | None
    label: str | None = None


@dataclass(frozen=True)
class ModelRanking:
    """What one representation returned, and what produced it."""

    name: str
    dimension: int
    provenance: dict[str, str] = field(default_factory=dict)
    results: tuple[RankedResult, ...] = ()


def _display_name(store: EmbeddingStore) -> str:
    return str(store.metadata.get("model") or store.metadata.get("backend") or "unknown")


class Catalog:
    """Several representations of one corpus, ranked by exact cosine similarity."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        stores: Sequence[EmbeddingStore],
        *,
        image_root: Path,
        projection: PcaProjection | None = None,
    ) -> None:
        if not stores:
            raise ValueError("at least one embedding store is required")
        self._require_one_corpus(stores)

        self._stores = tuple(stores)
        self._projection = projection
        path_by_id = {record.item_id: image_root / record.path for record in records}

        reference = stores[0]
        missing = set(reference.ids) - set(path_by_id)
        if missing:
            raise ValueError(
                f"manifest is missing {len(missing)} item(s) referenced by the "
                f"embedding stores, for example {sorted(missing)[0]!r}: a manifest "
                "missing corpus rows would degrade into silent 404s per tile "
                "instead of failing at startup"
            )
        self._path_by_id = path_by_id

        if projection is not None:
            self._validate_projection_matches_pca_store(projection)

        self._index_positions = [
            position for position, split in enumerate(reference.splits) if split == "index"
        ]
        if not self._index_positions:
            raise ValueError("embedding stores contain no index rows to search")
        if not any(split == "query" for split in reference.splits):
            raise ValueError(
                "embedding stores contain no query rows: the home page needs at "
                "least one query item to show by default"
            )
        self._index_ids = [reference.ids[position] for position in self._index_positions]
        self._label_by_id = {
            reference.ids[position]: reference.labels[position]
            for position in self._index_positions
        }
        self._label_by_query = dict(zip(reference.ids, reference.labels, strict=True))
        self._position_by_id = {value: position for position, value in enumerate(reference.ids)}
        self._indexes = [
            ExactCosineIndex(self._index_ids, store.vectors[self._index_positions])
            for store in stores
        ]

    @staticmethod
    def _require_one_corpus(stores: Sequence[EmbeddingStore]) -> None:
        """Refuse to compare models that did not rank the same corpus.

        A page built from mismatched stores would be meaningless and would look
        entirely correct, so this is a hard failure rather than a warning. A store
        with no recorded manifest hash is not comparable at all: without that, a
        missing hash on two stores would compare equal by omission and let a
        mismatched pair slip through.
        """

        for store in stores:
            digest = store.metadata.get("manifest_sha256")
            if not isinstance(digest, str):
                raise ValueError(
                    "supplied store has no recorded manifest hash: a store without a "
                    "recorded corpus identity cannot be shown to describe the same "
                    "corpus as the others"
                )

        reference = stores[0]
        reference_digest = reference.metadata.get("manifest_sha256")
        for store in stores[1:]:
            if (
                store.metadata.get("manifest_sha256") != reference_digest
                or store.ids != reference.ids
                or store.splits != reference.splits
                or store.labels != reference.labels
            ):
                raise ValueError(
                    "supplied stores did not rank the same corpus: their manifest hashes "
                    "or item orderings differ, so a side-by-side comparison would be "
                    "meaningless"
                )

    def _validate_projection_matches_pca_store(self, projection: PcaProjection) -> None:
        """Refuse a projection that was not fitted for the loaded PCA store.

        ``embeddings/encode.py`` runs these same two checks before embedding a
        query image on the CLI path. The served path must run them too: a
        projection whose component count happens to match the store's dimension
        would otherwise be accepted, and every upload would be projected through
        the wrong basis and ranked against vectors it was never fitted to compare.
        """

        position = self._pca_position()
        if position is None:
            return
        store = self._stores[position]
        recorded_size = store.metadata.get("image_size")
        if recorded_size is not None and int(recorded_size) != projection.image_size:
            raise ValueError(
                f"projection image_size {projection.image_size} does not match the "
                f"pca store's recorded image_size {int(recorded_size)}"
            )
        if projection.dimension != store.vectors.shape[1]:
            raise ValueError(
                f"projection produces {projection.dimension} dimensions but the pca "
                f"store holds {store.vectors.shape[1]}"
            )

    @classmethod
    def load(
        cls,
        *,
        manifest: Path,
        image_root: Path,
        stores: Sequence[Path],
        projection: Path | None,
    ) -> Catalog:
        for path in (manifest, *stores):
            if not path.is_file():
                raise ValueError(
                    f"missing input: {path}. Prepare the benchmark and embeddings first; "
                    "see docs/benchmark-eurosat.md"
                )
        loaded = [EmbeddingStore.load(path) for path in stores]
        cls._require_one_corpus(loaded)
        manifest_digest = file_sha256(manifest)
        stores_digest = loaded[0].metadata.get("manifest_sha256")
        if manifest_digest != stores_digest:
            raise ValueError(
                f"--manifest {manifest} does not match the manifest the supplied "
                f"stores were built from (manifest hash {manifest_digest}, stores "
                f"record {stores_digest}): a mismatched manifest would render the "
                "wrong images beside correct scores under a provenance line that "
                "does not describe what is shown"
            )
        basis = None
        if projection is not None:
            if not projection.is_file():
                raise ValueError(f"missing PCA projection: {projection}")
            basis = PcaProjection.load(projection)
        return cls(
            read_jsonl(manifest), loaded, image_root=image_root, projection=basis
        )

    @property
    def query_ids(self) -> tuple[str, ...]:
        reference = self._stores[0]
        return tuple(
            identifier
            for identifier, split in zip(reference.ids, reference.splits, strict=True)
            if split == "query"
        )

    @property
    def upload_available(self) -> bool:
        return self._projection is not None and self._pca_position() is not None

    @property
    def image_size(self) -> int:
        return self._projection.image_size if self._projection is not None else 0

    @property
    def index_size(self) -> int:
        return len(self._index_ids)

    def image_path(self, item_id: str) -> Path:
        try:
            return self._path_by_id[item_id]
        except KeyError as error:
            raise KeyError(f"unknown item: {item_id}") from error

    def label(self, item_id: str) -> str | None:
        return self._label_by_query[item_id]

    def _pca_position(self) -> int | None:
        for position, store in enumerate(self._stores):
            if store.metadata.get("backend") == PCA_BACKEND:
                return position
        return None

    def _provenance(self, store: EmbeddingStore) -> dict[str, str]:
        provenance = {
            "model": _display_name(store),
            "backend": str(store.metadata.get("backend", "unknown")),
            "index_items": str(len(self._index_ids)),
            "ranker": "exact-cosine",
        }
        for key in ("manifest_sha256", "checkpoint_sha256"):
            value = store.metadata.get(key)
            if isinstance(value, str):
                provenance[key] = value
        bands = store.metadata.get("bands")
        provenance["input"] = (
            ", ".join(str(band) for band in bands) if isinstance(bands, list) else "RGB"
        )
        return provenance

    def _rank(
        self,
        position: int,
        vector: NDArray[np.float32],
        *,
        k: int,
        exclude_id: str | None,
        query_label: str | None,
    ) -> ModelRanking:
        store = self._stores[position]
        found = self._indexes[position].search(vector, k=k, exclude_id=exclude_id)
        results = tuple(
            RankedResult(
                item_id=item.item_id,
                score=item.score,
                label=self._label_by_id.get(item.item_id),
                relevant=(
                    None
                    if query_label is None
                    else self._label_by_id.get(item.item_id) == query_label
                ),
            )
            for item in found
        )
        return ModelRanking(
            name=_display_name(store),
            dimension=int(store.vectors.shape[1]),
            provenance=self._provenance(store),
            results=results,
        )

    def rank_item(self, item_id: str, *, k: int) -> list[ModelRanking]:
        """Rank one corpus item with every supplied representation."""

        if item_id not in self._position_by_id:
            raise KeyError(f"unknown item: {item_id}")
        position = self._position_by_id[item_id]
        label = self._label_by_query.get(item_id)
        return [
            self._rank(
                index,
                store.vectors[position],
                k=k,
                exclude_id=item_id,
                query_label=label,
            )
            for index, store in enumerate(self._stores)
        ]

    def rank_uploaded(self, pixels: NDArray[np.float32], *, k: int) -> ModelRanking:
        """Rank a new image through the persisted PCA basis.

        Only PCA is available: it is the one representation fitted inside this
        project, so its basis can be reloaded without a model framework.
        """

        position = self._pca_position()
        if self._projection is None or position is None:
            raise ValueError(
                "no PCA projection is available: uploads need a PCA store and its "
                "saved projection; start the server with --projection"
            )
        vector = self._projection.transform(pixels)[0]
        return self._rank(position, vector, k=k, exclude_id=None, query_label=None)
