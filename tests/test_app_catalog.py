"""Ranking, provenance, and the invariant that makes a comparison meaningful."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from eo_visual_retrieval.app.catalog import Catalog
from eo_visual_retrieval.embeddings.projection import PcaProjection
from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.manifests import write_jsonl
from eo_visual_retrieval.models import ImageRecord, Split

IMAGE_SIZE = 8


def _records() -> list[ImageRecord]:
    rows: list[tuple[str, Split, str]] = [
        ("forest/a.png", "index", "forest"),
        ("forest/b.png", "index", "forest"),
        ("water/c.png", "index", "water"),
        ("forest/q.png", "query", "forest"),
    ]
    return [
        # Distinct per-record hashes: manifests.py rejects identical content hashes
        # split across index/query, and these four rows describe different images.
        ImageRecord(item_id=i, path=i, split=s, label=lab, metadata={"sha256": f"{n:064x}"})
        for n, (i, s, lab) in enumerate(rows)
    ]


def _store(
    records: list[ImageRecord], backend: str, model: str | None, *, manifest_sha256: str
) -> EmbeddingStore:
    vectors = np.asarray(
        [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.99, 0.01]], dtype=np.float32
    )
    metadata: dict[str, object] = {
        "backend": backend,
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": "b" * 64,
    }
    if model is not None:
        metadata["model"] = model
    return EmbeddingStore(
        ids=tuple(r.item_id for r in records),
        vectors=vectors,
        labels=tuple(r.label for r in records),
        splits=tuple(r.split for r in records),
        metadata=metadata,
    )


def _write(tmp_path: Path, records: list[ImageRecord]) -> Path:
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(records, manifest)
    return manifest


def _digest(manifest: Path) -> str:
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _catalog(tmp_path: Path) -> Catalog:
    records = _records()
    manifest = _write(tmp_path, records)
    digest = _digest(manifest)
    paths = []
    for backend, model in (("pca", None), ("dinov2", "dinov2_vits14")):
        path = tmp_path / f"{backend}.npz"
        _store(records, backend, model, manifest_sha256=digest).save(path)
        paths.append(path)
    return Catalog.load(manifest=manifest, image_root=tmp_path, stores=paths, projection=None)


def test_catalog_ranks_a_corpus_query_with_every_supplied_model(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    rankings = catalog.rank_item("forest/q.png", k=2)

    assert [r.name for r in rankings] == ["pca", "dinov2_vits14"]
    for ranking in rankings:
        assert len(ranking.results) == 2
        # The query is a query-split row, so it is not among the index results.
        assert "forest/q.png" not in [x.item_id for x in ranking.results]
        assert ranking.results[0].item_id == "forest/a.png"
        assert ranking.results[0].relevant is True


def test_display_name_falls_back_to_backend_when_no_model_is_recorded(
    tmp_path: Path,
) -> None:
    """The published PCA store records model=None, so this fallback is load-bearing."""
    catalog = _catalog(tmp_path)

    assert catalog.rank_item("forest/q.png", k=1)[0].name == "pca"


def test_provenance_reports_what_produced_the_ranking(tmp_path: Path) -> None:
    records = _records()
    manifest = _write(tmp_path, records)
    digest = _digest(manifest)
    paths = []
    for backend, model in (("pca", None), ("dinov2", "dinov2_vits14")):
        path = tmp_path / f"{backend}.npz"
        _store(records, backend, model, manifest_sha256=digest).save(path)
        paths.append(path)
    catalog = Catalog.load(manifest=manifest, image_root=tmp_path, stores=paths, projection=None)

    provenance = catalog.rank_item("forest/q.png", k=1)[0].provenance

    assert provenance["manifest_sha256"] == digest
    assert provenance["checkpoint_sha256"] == "b" * 64
    assert provenance["index_items"] == "3"
    assert provenance["ranker"] == "exact-cosine"


def test_catalog_refuses_stores_that_ranked_different_corpora(tmp_path: Path) -> None:
    """Comparing models over different corpora is meaningless and looks fine on screen."""
    records = _records()
    manifest = _write(tmp_path, records)
    digest = _digest(manifest)
    first = tmp_path / "first.npz"
    _store(records, "pca", None, manifest_sha256=digest).save(first)

    mismatched = _store(records, "dinov2", "dinov2_vits14", manifest_sha256=digest)
    second = tmp_path / "second.npz"
    EmbeddingStore(
        ids=mismatched.ids,
        vectors=mismatched.vectors,
        labels=mismatched.labels,
        splits=mismatched.splits,
        metadata={**mismatched.metadata, "manifest_sha256": "c" * 64},
    ).save(second)

    with pytest.raises(ValueError, match="did not rank the same corpus"):
        Catalog.load(
            manifest=manifest, image_root=tmp_path, stores=[first, second], projection=None
        )


def test_catalog_refuses_stores_whose_ids_disagree(tmp_path: Path) -> None:
    records = _records()
    manifest = _write(tmp_path, records)
    digest = _digest(manifest)
    first = tmp_path / "first.npz"
    _store(records, "pca", None, manifest_sha256=digest).save(first)

    reordered = _store(records, "dinov2", "dinov2_vits14", manifest_sha256=digest)
    second = tmp_path / "second.npz"
    EmbeddingStore(
        ids=tuple(reversed(reordered.ids)),
        vectors=reordered.vectors,
        labels=tuple(reversed(reordered.labels)),
        splits=tuple(reversed(reordered.splits)),
        metadata=reordered.metadata,
    ).save(second)

    with pytest.raises(ValueError, match="did not rank the same corpus"):
        Catalog.load(
            manifest=manifest, image_root=tmp_path, stores=[first, second], projection=None
        )


def test_catalog_refuses_a_store_with_no_recorded_corpus_identity(tmp_path: Path) -> None:
    """A store that never recorded a manifest hash can't be shown to match another."""
    records = _records()
    manifest = _write(tmp_path, records)
    digest = _digest(manifest)
    first = tmp_path / "first.npz"
    _store(records, "pca", None, manifest_sha256=digest).save(first)

    no_identity = _store(records, "dinov2", "dinov2_vits14", manifest_sha256=digest)
    metadata = dict(no_identity.metadata)
    del metadata["manifest_sha256"]
    second = tmp_path / "second.npz"
    EmbeddingStore(
        ids=no_identity.ids,
        vectors=no_identity.vectors,
        labels=no_identity.labels,
        splits=no_identity.splits,
        metadata=metadata,
    ).save(second)

    with pytest.raises(ValueError, match="no recorded manifest hash"):
        Catalog.load(
            manifest=manifest, image_root=tmp_path, stores=[first, second], projection=None
        )


def test_rank_uploaded_ranks_through_the_projection_without_relevance(tmp_path: Path) -> None:
    """An uploaded image carries no label, so every result must read relevant=None."""
    records = _records()
    manifest = _write(tmp_path, records)
    digest = _digest(manifest)
    paths = []
    for backend, model in (("pca", None), ("dinov2", "dinov2_vits14")):
        path = tmp_path / f"{backend}.npz"
        _store(records, backend, model, manifest_sha256=digest).save(path)
        paths.append(path)

    features = IMAGE_SIZE * IMAGE_SIZE * 3
    components = np.zeros((2, features), dtype=np.float32)
    components[0, 0] = 1.0
    components[1, 1] = 1.0
    projection = PcaProjection(
        mean=np.zeros(features, dtype=np.float32),
        components=components,
        image_size=IMAGE_SIZE,
        seed=0,
    )
    projection_path = tmp_path / "projection.npz"
    projection.save(projection_path)

    catalog = Catalog.load(
        manifest=manifest, image_root=tmp_path, stores=paths, projection=projection_path
    )

    pixels = np.zeros((1, features), dtype=np.float32)
    pixels[0, 0] = 1.0
    pixels[0, 1] = 0.5

    ranking = catalog.rank_uploaded(pixels, k=2)

    assert ranking.name == "pca"
    assert len(ranking.results) == 2
    assert all(result.relevant is None for result in ranking.results)


def test_upload_is_unavailable_without_a_projection(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    assert catalog.upload_available is False
    with pytest.raises(ValueError, match="no PCA projection"):
        catalog.rank_uploaded(np.zeros((1, IMAGE_SIZE * IMAGE_SIZE * 3), dtype=np.float32), k=1)


def test_catalog_load_refuses_a_manifest_that_does_not_match_the_stores(tmp_path: Path) -> None:
    """--manifest must be the exact file the supplied stores were built from.

    Nothing else proves that: the stores can agree with each other perfectly and
    still be paired with the wrong manifest, which would render the wrong images
    beside correct scores under a provenance line that lies about what is shown.
    """
    records = _records()
    manifest = _write(tmp_path, records)
    digest = _digest(manifest)
    paths = []
    for backend, model in (("pca", None), ("dinov2", "dinov2_vits14")):
        path = tmp_path / f"{backend}.npz"
        _store(records, backend, model, manifest_sha256=digest).save(path)
        paths.append(path)

    # A different manifest file at the same path: same items plus one extra row,
    # so its bytes -- and therefore its sha256 -- differ from what the stores
    # recorded.
    extra_record = ImageRecord(
        item_id="forest/extra.png",
        path="forest/extra.png",
        split="index",
        label="forest",
        metadata={"sha256": "9" * 64},
    )
    write_jsonl([*records, extra_record], manifest)
    assert _digest(manifest) != digest

    with pytest.raises(ValueError, match="does not match the manifest"):
        Catalog.load(manifest=manifest, image_root=tmp_path, stores=paths, projection=None)


def test_catalog_refuses_a_manifest_missing_rows_the_stores_reference(tmp_path: Path) -> None:
    """A manifest missing a corpus row must fail at startup, not 404 per tile."""
    records = _records()
    manifest = _write(tmp_path, records)
    digest = _digest(manifest)
    paths = []
    for backend, model in (("pca", None), ("dinov2", "dinov2_vits14")):
        path = tmp_path / f"{backend}.npz"
        _store(records, backend, model, manifest_sha256=digest).save(path)
        paths.append(path)
    loaded_stores = [EmbeddingStore.load(path) for path in paths]

    incomplete_records = records[:-1]  # drop a row the stores still reference

    with pytest.raises(ValueError, match="manifest is missing"):
        Catalog(incomplete_records, loaded_stores, image_root=tmp_path, projection=None)


def test_catalog_refuses_a_projection_with_mismatched_image_size(tmp_path: Path) -> None:
    """Mirrors the image_size check embeddings/encode.py runs on the CLI path."""
    records = _records()
    manifest = _write(tmp_path, records)
    digest = _digest(manifest)
    paths = []
    for backend, model in (("pca", None), ("dinov2", "dinov2_vits14")):
        path = tmp_path / f"{backend}.npz"
        store = _store(records, backend, model, manifest_sha256=digest)
        if backend == "pca":
            store = EmbeddingStore(
                ids=store.ids,
                vectors=store.vectors,
                labels=store.labels,
                splits=store.splits,
                metadata={**store.metadata, "image_size": IMAGE_SIZE},
            )
        store.save(path)
        paths.append(path)

    other_size = IMAGE_SIZE + 1
    features = other_size * other_size * 3
    projection = PcaProjection(
        mean=np.zeros(features, dtype=np.float32),
        components=np.zeros((2, features), dtype=np.float32),
        image_size=other_size,
        seed=0,
    )
    projection_path = tmp_path / "projection.npz"
    projection.save(projection_path)

    with pytest.raises(ValueError, match="image_size"):
        Catalog.load(
            manifest=manifest, image_root=tmp_path, stores=paths, projection=projection_path
        )


def test_catalog_refuses_a_projection_with_mismatched_dimension(tmp_path: Path) -> None:
    """A basis with the right image_size but the wrong component count is still wrong.

    An upload projected through it would be ranked against a pca store it was
    never fitted to compare, silently, because the component count alone gives
    no visible symptom.
    """
    records = _records()
    manifest = _write(tmp_path, records)
    digest = _digest(manifest)
    paths = []
    for backend, model in (("pca", None), ("dinov2", "dinov2_vits14")):
        path = tmp_path / f"{backend}.npz"
        _store(records, backend, model, manifest_sha256=digest).save(path)
        paths.append(path)

    features = IMAGE_SIZE * IMAGE_SIZE * 3
    # The pca store's vectors are 2-dimensional; three components disagrees.
    projection = PcaProjection(
        mean=np.zeros(features, dtype=np.float32),
        components=np.zeros((3, features), dtype=np.float32),
        image_size=IMAGE_SIZE,
        seed=0,
    )
    projection_path = tmp_path / "projection.npz"
    projection.save(projection_path)

    with pytest.raises(ValueError, match="dimensions"):
        Catalog.load(
            manifest=manifest, image_root=tmp_path, stores=paths, projection=projection_path
        )


def test_catalog_refuses_stores_with_no_query_rows(tmp_path: Path) -> None:
    """home() indexes query_ids[0]; an all-index corpus must fail at startup."""
    rows: list[tuple[str, Split, str]] = [
        ("forest/a.png", "index", "forest"),
        ("forest/b.png", "index", "forest"),
        ("water/c.png", "index", "water"),
    ]
    records = [
        ImageRecord(item_id=i, path=i, split=s, label=lab, metadata={"sha256": f"{n:064x}"})
        for n, (i, s, lab) in enumerate(rows)
    ]
    manifest = _write(tmp_path, records)
    store = EmbeddingStore(
        ids=tuple(r.item_id for r in records),
        vectors=np.asarray([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]], dtype=np.float32),
        labels=tuple(r.label for r in records),
        splits=tuple(r.split for r in records),
        metadata={"backend": "pca", "manifest_sha256": _digest(manifest)},
    )
    path = tmp_path / "pca.npz"
    store.save(path)

    with pytest.raises(ValueError, match="no query rows"):
        Catalog.load(manifest=manifest, image_root=tmp_path, stores=[path], projection=None)


def test_query_ids_are_the_query_split_only(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    assert catalog.query_ids == ("forest/q.png",)


def test_unknown_item_is_rejected(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    with pytest.raises(KeyError, match="absent.png"):
        catalog.rank_item("absent.png", k=1)
