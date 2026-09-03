# Product Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve a representation-comparison interface over local EuroSAT v1 stores, where one query is ranked side by side by every supplied model with full provenance, and an uploaded RGB image is ranked through the persisted PCA basis.

**Architecture:** A new optional `app` extra under `src/eo_visual_retrieval/app/`. `catalog.py` holds all ranking and provenance logic and imports nothing web-related, so it is testable without a server. `thumbnails.py` converts GeoTIFF to JPEG because browsers cannot render GeoTIFF. `uploads.py` treats incoming bytes as hostile. `main.py` plus one Jinja template is a thin routing layer. No PyTorch is imported by the served process.

**Tech Stack:** Python 3.11/3.12, FastAPI, Uvicorn, Jinja2, python-multipart, NumPy, Pillow, pytest, Ruff, Mypy.

## Global Constraints

- Line length 100; Ruff `select = ["E", "F", "I", "UP", "B"]` must pass.
- Mypy runs over `src`, `tests`, `scripts` with `disallow_untyped_defs`; every function needs annotations.
- Coverage must stay at or above 75%.
- No published result may change. `docs/results/*.json` and `docs/assets/*.png` are not edited by this plan.
- `src/eo_visual_retrieval/evaluation.py` and `retrieval.py` must stay byte-identical.
- `docs/validation.md` records only executed evidence; never write a number a run did not produce.
- Never commit anything under `data/`, `artifacts/`, or `outputs/` — all are gitignored.
- Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Local environment: `C:\Users\nikos\.venvs\eovr\Scripts\python.exe`, run from the repo root.
- The served process must not import `torch`, `torchvision`, or `terratorch`. A test enforces this.
- Display name for a representation is `metadata["model"] or metadata["backend"] or "unknown"`, matching `visualization.py`. The PCA store records `model=None`, so the fallback is load-bearing, not defensive.

## File Structure

| File | Responsibility |
|---|---|
| `src/eo_visual_retrieval/app/__init__.py` | **Create.** Package marker. |
| `src/eo_visual_retrieval/app/catalog.py` | **Create.** Load stores and manifest, enforce the comparability invariant, rank, expose provenance. No web imports. |
| `src/eo_visual_retrieval/app/thumbnails.py` | **Create.** GeoTIFF to bounded JPEG bytes, LRU-cached. |
| `src/eo_visual_retrieval/app/uploads.py` | **Create.** Validate and decode untrusted bytes, or refuse. |
| `src/eo_visual_retrieval/app/main.py` | **Create.** FastAPI app factory and routes. |
| `src/eo_visual_retrieval/app/templates/compare.html` | **Create.** The comparison page. |
| `src/eo_visual_retrieval/cli.py` | **Modify.** Add the `serve` subcommand before `return parser` at line 684. |
| `pyproject.toml` | **Modify.** Add the `app` extra. |
| `tests/test_app_catalog.py` | **Create.** Invariant, ranking, provenance, upload dispatch. |
| `tests/test_app_thumbnails.py` | **Create.** Conversion, bounds, caching. |
| `tests/test_app_uploads.py` | **Create.** Size cap, undecodable input, pixel bomb. |
| `tests/test_app_routes.py` | **Create.** Routes via `TestClient`, and the no-torch guarantee. |

---

### Task 1: The catalog and its comparability invariant

**Files:**
- Create: `src/eo_visual_retrieval/app/__init__.py`, `src/eo_visual_retrieval/app/catalog.py`
- Modify: `pyproject.toml` (add the `app` extra)
- Test: `tests/test_app_catalog.py`

**Interfaces:**
- Consumes: `EmbeddingStore` from `eo_visual_retrieval.embeddings.store`, `ExactCosineIndex` from `eo_visual_retrieval.retrieval`, `read_jsonl` from `eo_visual_retrieval.manifests`, `PcaProjection` from `eo_visual_retrieval.embeddings.projection`.
- Produces:
  - `RankedResult` frozen dataclass: `item_id: str`, `score: float`, `relevant: bool | None`.
  - `ModelRanking` frozen dataclass: `name: str`, `backend: str`, `dimension: int`, `provenance: dict[str, str]`, `results: tuple[RankedResult, ...]`.
  - `Catalog` class with `load(manifest: Path, stores: Sequence[Path], projection: Path | None) -> Catalog`, `query_ids: tuple[str, ...]`, `image_path(item_id: str) -> Path`, `rank_item(item_id: str, *, k: int) -> list[ModelRanking]`, `rank_uploaded(pixels: NDArray[np.float32], *, k: int) -> ModelRanking`, `upload_available: bool`, `image_size: int`.

- [ ] **Step 1: Add the `app` extra**

In `pyproject.toml`, immediately after the `search = [...]` block, add:

```toml
app = [
  "fastapi>=0.115,<1",
  "uvicorn>=0.30,<1",
  "jinja2>=3.1,<4",
  "python-multipart>=0.0.9,<1",
]
```

Then install it locally:

```bash
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pip install -e ".[app]"
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_app_catalog.py`:

```python
"""Ranking, provenance, and the invariant that makes a comparison meaningful."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eo_visual_retrieval.app.catalog import Catalog
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
        ImageRecord(item_id=i, path=i, split=s, label=lab, metadata={"sha256": "0" * 64})
        for i, s, lab in rows
    ]


def _store(records: list[ImageRecord], backend: str, model: str | None) -> EmbeddingStore:
    vectors = np.asarray(
        [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.99, 0.01]], dtype=np.float32
    )
    metadata: dict[str, object] = {
        "backend": backend,
        "manifest_sha256": "a" * 64,
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


def _catalog(tmp_path: Path) -> Catalog:
    records = _records()
    manifest = _write(tmp_path, records)
    paths = []
    for backend, model in (("pca", None), ("dinov2", "dinov2_vits14")):
        path = tmp_path / f"{backend}.npz"
        _store(records, backend, model).save(path)
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
    catalog = _catalog(tmp_path)

    provenance = catalog.rank_item("forest/q.png", k=1)[0].provenance

    assert provenance["manifest_sha256"] == "a" * 64
    assert provenance["checkpoint_sha256"] == "b" * 64
    assert provenance["index_items"] == "3"
    assert provenance["ranker"] == "exact-cosine"


def test_catalog_refuses_stores_that_ranked_different_corpora(tmp_path: Path) -> None:
    """Comparing models over different corpora is meaningless and looks fine on screen."""
    records = _records()
    manifest = _write(tmp_path, records)
    first = tmp_path / "first.npz"
    _store(records, "pca", None).save(first)

    mismatched = _store(records, "dinov2", "dinov2_vits14")
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
    first = tmp_path / "first.npz"
    _store(records, "pca", None).save(first)

    reordered = _store(records, "dinov2", "dinov2_vits14")
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


def test_upload_is_unavailable_without_a_projection(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    assert catalog.upload_available is False
    with pytest.raises(ValueError, match="no PCA projection"):
        catalog.rank_uploaded(np.zeros((1, IMAGE_SIZE * IMAGE_SIZE * 3), dtype=np.float32), k=1)


def test_query_ids_are_the_query_split_only(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    assert catalog.query_ids == ("forest/q.png",)


def test_unknown_item_is_rejected(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    with pytest.raises(KeyError, match="absent.png"):
        catalog.rank_item("absent.png", k=1)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest tests/test_app_catalog.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eo_visual_retrieval.app'`

- [ ] **Step 4: Write the module**

Create `src/eo_visual_retrieval/app/__init__.py`:

```python
"""Served comparison surface over prepared embedding stores."""
```

Create `src/eo_visual_retrieval/app/catalog.py`:

```python
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


@dataclass(frozen=True)
class ModelRanking:
    """What one representation returned, and what produced it."""

    name: str
    backend: str
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

        self._records = tuple(records)
        self._stores = tuple(stores)
        self._image_root = image_root
        self._projection = projection
        self._path_by_id = {record.item_id: image_root / record.path for record in records}

        reference = stores[0]
        self._index_positions = [
            position for position, split in enumerate(reference.splits) if split == "index"
        ]
        if not self._index_positions:
            raise ValueError("embedding stores contain no index rows to search")
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
        entirely correct, so this is a hard failure rather than a warning.
        """

        reference = stores[0]
        digest = reference.metadata.get("manifest_sha256")
        for store in stores[1:]:
            if store.metadata.get("manifest_sha256") != digest or store.ids != reference.ids:
                raise ValueError(
                    "supplied stores did not rank the same corpus: their manifest hashes "
                    "or item orderings differ, so a side-by-side comparison would be "
                    "meaningless"
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

    def image_path(self, item_id: str) -> Path:
        try:
            return self._path_by_id[item_id]
        except KeyError as error:
            raise KeyError(f"unknown item: {item_id}") from error

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
            backend=str(store.metadata.get("backend", "unknown")),
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
                "uploads need a PCA store and its saved projection; start the server "
                "with --projection"
            )
        vector = self._projection.transform(pixels)[0]
        return self._rank(position, vector, k=k, exclude_id=None, query_label=None)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest tests/test_app_catalog.py -q`
Expected: PASS, 8 tests

- [ ] **Step 6: Run the full gates**

```bash
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m ruff check .
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m mypy
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest -q --cov=eo_visual_retrieval --cov-report=term --cov-fail-under=75
```

Expected: all clean, coverage at or above 75%.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/eo_visual_retrieval/app tests/test_app_catalog.py
git commit -m "$(cat <<'EOF'
Add the served catalog and its comparability invariant

Ranking and provenance live in a module that imports nothing web-related and no
model framework, so what the page claims is testable without a server.

The catalog refuses to load stores whose manifest hashes or item orderings
differ. Comparing representations that ranked different corpora would be
meaningless and would look entirely correct on screen, so it fails at load.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Thumbnails

Browsers cannot render GeoTIFF, so this is a functional requirement rather than presentation polish.

**Files:**
- Create: `src/eo_visual_retrieval/app/thumbnails.py`
- Test: `tests/test_app_thumbnails.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `thumbnail_jpeg(path: Path, *, size: int = 128, quality: int = 85) -> bytes`, and `clear_thumbnail_cache() -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_app_thumbnails.py`:

```python
"""GeoTIFF to browser-renderable JPEG, bounded and cached."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eo_visual_retrieval.app.thumbnails import clear_thumbnail_cache, thumbnail_jpeg


def _tiff(path: Path, size: int = 64) -> Path:
    pixels = np.random.default_rng(3).integers(0, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, format="TIFF")
    return path


def test_geotiff_becomes_a_decodable_jpeg(tmp_path: Path) -> None:
    data = thumbnail_jpeg(_tiff(tmp_path / "chip.tif"), size=32)

    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "JPEG"
        assert max(image.size) <= 32


def test_thumbnails_are_cached_by_path_and_size(tmp_path: Path) -> None:
    clear_thumbnail_cache()
    path = _tiff(tmp_path / "chip.tif")

    first = thumbnail_jpeg(path, size=32)
    second = thumbnail_jpeg(path, size=32)

    assert first is second, "repeat views must not re-decode the source raster"


def test_missing_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        thumbnail_jpeg(tmp_path / "absent.tif")


def test_size_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        thumbnail_jpeg(_tiff(tmp_path / "chip.tif"), size=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest tests/test_app_thumbnails.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eo_visual_retrieval.app.thumbnails'`

- [ ] **Step 3: Write the module**

Create `src/eo_visual_retrieval/app/thumbnails.py`:

```python
"""Render source rasters as JPEG for the browser.

The corpus is stored as GeoTIFF, which no browser renders, so conversion is a
functional requirement. Results are cached because a comparison page requests
the same tiles repeatedly and decoding a raster per view would dominate its cost.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

from PIL import Image

CACHE_ENTRIES = 4096


@lru_cache(maxsize=CACHE_ENTRIES)
def _render(path: str, size: int, quality: int) -> bytes:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.BICUBIC)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def thumbnail_jpeg(path: Path, *, size: int = 128, quality: int = 85) -> bytes:
    """Return JPEG bytes for one source raster."""

    if size < 1:
        raise ValueError("size must be positive")
    if not 1 <= quality <= 95:
        raise ValueError("quality must be between 1 and 95")
    if not path.is_file():
        raise ValueError(f"source image does not exist: {path}")
    return _render(str(path), size, quality)


def clear_thumbnail_cache() -> None:
    """Drop cached renders. Used by tests and after a corpus changes on disk."""

    _render.cache_clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest tests/test_app_thumbnails.py -q`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add src/eo_visual_retrieval/app/thumbnails.py tests/test_app_thumbnails.py
git commit -m "$(cat <<'EOF'
Render corpus rasters as JPEG for the browser

The corpus is GeoTIFF, which browsers do not render, so this is a functional
requirement rather than presentation polish. Renders are cached because a
comparison page requests the same tiles repeatedly.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Untrusted uploads

**Files:**
- Create: `src/eo_visual_retrieval/app/uploads.py`
- Test: `tests/test_app_uploads.py`

**Interfaces:**
- Consumes: `load_flat_rgb`-equivalent behaviour, but implemented here against bytes rather than paths.
- Produces: `MAX_UPLOAD_BYTES: int`, `MAX_UPLOAD_PIXELS: int`, `decode_upload(data: bytes, *, image_size: int) -> NDArray[np.float32]` returning shape `(1, image_size * image_size * 3)` scaled to 0–1, raising `ValueError` on any refusal.

- [ ] **Step 1: Write the failing test**

Create `tests/test_app_uploads.py`:

```python
"""Uploaded bytes are hostile until proven otherwise."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from eo_visual_retrieval.app.uploads import (
    MAX_UPLOAD_BYTES,
    decode_upload,
)

IMAGE_SIZE = 8


def _png(width: int = 32, height: int = 32) -> bytes:
    pixels = np.random.default_rng(5).integers(0, 255, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def test_valid_upload_becomes_flat_scaled_pixels() -> None:
    actual = decode_upload(_png(), image_size=IMAGE_SIZE)

    assert actual.shape == (1, IMAGE_SIZE * IMAGE_SIZE * 3)
    assert actual.dtype == np.float32
    assert float(actual.min()) >= 0.0 and float(actual.max()) <= 1.0


def test_oversize_upload_is_refused_before_decoding() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        decode_upload(b"\x00" * (MAX_UPLOAD_BYTES + 1), image_size=IMAGE_SIZE)


def test_empty_upload_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        decode_upload(b"", image_size=IMAGE_SIZE)


def test_undecodable_bytes_are_refused() -> None:
    """Type is established by decoding, not by trusting a declared content type."""
    with pytest.raises(ValueError, match="not a readable image"):
        decode_upload(b"this is not an image at all", image_size=IMAGE_SIZE)


def test_declared_pixel_bomb_is_refused() -> None:
    # A tiny file that declares enormous dimensions is the classic bomb shape.
    header = io.BytesIO()
    Image.new("RGB", (2, 2)).save(header, format="PNG")
    payload = bytearray(header.getvalue())
    # Corrupt the IHDR width/height to advertise a huge canvas.
    payload[16:24] = (60000).to_bytes(4, "big") + (60000).to_bytes(4, "big")

    with pytest.raises(ValueError):
        decode_upload(bytes(payload), image_size=IMAGE_SIZE)


def test_image_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="image_size"):
        decode_upload(_png(), image_size=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest tests/test_app_uploads.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eo_visual_retrieval.app.uploads'`

- [ ] **Step 3: Write the module**

Create `src/eo_visual_retrieval/app/uploads.py`:

```python
"""Decode an uploaded image, or refuse it.

This surface is intended to be publicly reachable, so incoming bytes are treated
as hostile: they are size-capped before any decode is attempted, never written to
disk, and their type is established by decoding rather than by trusting a
declared content type.
"""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_UPLOAD_PIXELS = 64_000_000


def decode_upload(data: bytes, *, image_size: int) -> NDArray[np.float32]:
    """Return one flattened, 0-1 scaled RGB row from uploaded bytes."""

    if image_size < 1:
        raise ValueError("image_size must be positive")
    if not data:
        raise ValueError("uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"uploaded file exceeds the {MAX_UPLOAD_BYTES}-byte limit")

    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_UPLOAD_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            resized = source.convert("RGB").resize((image_size, image_size))
            pixels = np.asarray(resized, dtype=np.float32) / 255.0
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("uploaded file is not a readable image") from error
    finally:
        Image.MAX_IMAGE_PIXELS = previous

    return pixels.reshape(1, -1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest tests/test_app_uploads.py -q`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/eo_visual_retrieval/app/uploads.py tests/test_app_uploads.py
git commit -m "$(cat <<'EOF'
Decode uploads defensively, or refuse them

Bytes are capped before any decode is attempted, never written to disk, and
their type is established by decoding rather than by a declared content type.
Pillow's pixel ceiling is lowered around the decode to bound bombs.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Routes, template, and `eovr serve`

**Files:**
- Create: `src/eo_visual_retrieval/app/main.py`, `src/eo_visual_retrieval/app/templates/compare.html`
- Modify: `src/eo_visual_retrieval/cli.py` (add `serve` immediately before `return parser` at line 684)
- Test: `tests/test_app_routes.py`

**Interfaces:**
- Consumes: `Catalog`, `ModelRanking`, `RankedResult` from Task 1; `thumbnail_jpeg` from Task 2; `decode_upload`, `MAX_UPLOAD_BYTES` from Task 3.
- Produces: `create_app(catalog: Catalog, *, k: int = 5) -> FastAPI`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_app_routes.py`:

```python
"""Routes, and the guarantee that the served process carries no model framework."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eo_visual_retrieval.app.catalog import Catalog
from eo_visual_retrieval.embeddings.projection import PcaProjection
from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.manifests import write_jsonl
from eo_visual_retrieval.models import ImageRecord, Split

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from eo_visual_retrieval.app.main import create_app  # noqa: E402

IMAGE_SIZE = 8
FEATURES = IMAGE_SIZE * IMAGE_SIZE * 3


def _catalog(tmp_path: Path) -> Catalog:
    rows: list[tuple[str, Split, str]] = [
        ("forest/a.tif", "index", "forest"),
        ("forest/b.tif", "index", "forest"),
        ("water/c.tif", "index", "water"),
        ("forest/q.tif", "query", "forest"),
    ]
    records = [
        ImageRecord(item_id=i, path=i, split=s, label=lab, metadata={"sha256": "0" * 64})
        for i, s, lab in rows
    ]
    for record in records:
        destination = tmp_path / record.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 60, dtype=np.uint8)
        ).save(destination, format="TIFF")

    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(records, manifest)

    store = EmbeddingStore(
        ids=tuple(r.item_id for r in records),
        vectors=np.asarray(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.99, 0.01]], dtype=np.float32
        ),
        labels=tuple(r.label for r in records),
        splits=tuple(r.split for r in records),
        metadata={"backend": "pca", "manifest_sha256": "a" * 64, "image_size": IMAGE_SIZE},
    )
    store_path = tmp_path / "pca.npz"
    store.save(store_path)

    projection_path = tmp_path / "projection.npz"
    PcaProjection(
        mean=np.zeros(FEATURES, dtype=np.float32),
        components=np.eye(2, FEATURES, dtype=np.float32) + 0.5,
        image_size=IMAGE_SIZE,
        seed=42,
    ).save(projection_path)

    return Catalog.load(
        manifest=manifest,
        image_root=tmp_path,
        stores=[store_path],
        projection=projection_path,
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(_catalog(tmp_path), k=2))


def test_index_page_renders_a_comparison(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "exact-cosine" in response.text


def test_comparison_names_the_model_and_its_provenance(client: TestClient) -> None:
    response = client.get("/compare", params={"item_id": "forest/q.tif"})

    assert response.status_code == 200
    assert "pca" in response.text
    assert "a" * 64 in response.text


def test_unknown_item_is_not_found(client: TestClient) -> None:
    response = client.get("/compare", params={"item_id": "absent.tif"})

    assert response.status_code == 404


def test_thumbnail_route_returns_jpeg(client: TestClient) -> None:
    response = client.get("/thumbnail", params={"item_id": "forest/a.tif"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.format == "JPEG"


def test_thumbnail_refuses_an_item_outside_the_corpus(client: TestClient) -> None:
    """The catalog is the only source of paths, so traversal cannot reach disk."""
    response = client.get("/thumbnail", params={"item_id": "../../etc/passwd"})

    assert response.status_code == 404


def test_upload_is_ranked_without_relevance_colouring(client: TestClient) -> None:
    buffer = io.BytesIO()
    Image.fromarray(
        np.full((16, 16, 3), 90, dtype=np.uint8)
    ).save(buffer, format="PNG")

    response = client.post(
        "/compare/upload", files={"image": ("chip.png", buffer.getvalue(), "image/png")}
    )

    assert response.status_code == 200
    assert "no label" in response.text.lower()


def test_undecodable_upload_is_a_bad_request(client: TestClient) -> None:
    response = client.post(
        "/compare/upload", files={"image": ("x.png", b"not an image", "image/png")}
    )

    assert response.status_code == 400
    assert "not a readable image" in response.text


def test_served_process_imports_no_model_framework() -> None:
    """The deployable image stays small only if nothing pulls in a framework."""
    import eo_visual_retrieval.app.catalog  # noqa: F401
    import eo_visual_retrieval.app.main  # noqa: F401
    import eo_visual_retrieval.app.thumbnails  # noqa: F401
    import eo_visual_retrieval.app.uploads  # noqa: F401

    for forbidden in ("torch", "torchvision", "terratorch", "sklearn"):
        assert forbidden not in sys.modules, f"the served surface must not import {forbidden}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest tests/test_app_routes.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eo_visual_retrieval.app.main'`

- [ ] **Step 3: Write the template**

Create `src/eo_visual_retrieval/app/templates/compare.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EO visual retrieval — representation comparison</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
    h1 { font-size: 1.25rem; }
    .note { color: #555; font-size: 0.85rem; max-width: 46rem; }
    .row { margin: 1.5rem 0; border-top: 1px solid #ddd; padding-top: 1rem; }
    .tiles { display: flex; gap: 0.5rem; align-items: flex-start; }
    .tile img { width: 96px; height: 96px; display: block; border: 3px solid #ccc; }
    .query img { border-color: #2864dc; }
    .hit img { border-color: #239646; }
    .miss img { border-color: #cd3737; }
    .unknown img { border-color: #999; }
    .cap { font-size: 0.7rem; color: #444; }
    .prov { font-size: 0.7rem; color: #666; font-family: ui-monospace, monospace; }
    form { margin: 1rem 0; }
  </style>
</head>
<body>
  <h1>Representation comparison</h1>
  <p class="note">
    One query, ranked by each representation with exact cosine similarity over
    precomputed embeddings. Blue marks the query, green a same-class result, red a
    different-class result.
  </p>

  <form method="get" action="/compare">
    <label for="item_id">Corpus query</label>
    <select name="item_id" id="item_id">
      {% for candidate in query_ids %}
      <option value="{{ candidate }}" {% if candidate == item_id %}selected{% endif %}>{{ candidate }}</option>
      {% endfor %}
    </select>
    <button type="submit">Rank</button>
  </form>

  {% if upload_available %}
  <form method="post" action="/compare/upload" enctype="multipart/form-data">
    <label for="image">Or upload an image</label>
    <input type="file" name="image" id="image" accept="image/*" required>
    <button type="submit">Rank upload</button>
  </form>
  {% else %}
  <p class="note">Upload is disabled: start the server with <code>--projection</code> to enable it.</p>
  {% endif %}

  {% if is_upload %}
  <p class="note">
    This query was uploaded, so it carries <strong>no label</strong>. Results are
    shown without relevance colouring, and no per-query metric is computed: grey
    means unknown, not wrong. Uploads are embedded with PCA only, the one
    representation this project fits itself.
  </p>
  {% endif %}

  {% for ranking in rankings %}
  <div class="row">
    <div><strong>{{ ranking.name }}</strong> &mdash; {{ ranking.dimension }} dimensions</div>
    <div class="prov">
      {% for key, value in ranking.provenance.items() %}{{ key }}={{ value }} {% endfor %}
    </div>
    <div class="tiles">
      {% if query_id %}
      <div class="tile query">
        <img src="/thumbnail?item_id={{ query_id | urlencode }}" alt="query">
        <div class="cap">query</div>
      </div>
      {% endif %}
      {% for result in ranking.results %}
      <div class="tile {% if result.relevant is none %}unknown{% elif result.relevant %}hit{% else %}miss{% endif %}">
        <img src="/thumbnail?item_id={{ result.item_id | urlencode }}" alt="result">
        <div class="cap">#{{ loop.index }} {{ "%.3f"|format(result.score) }}</div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endfor %}
</body>
</html>
```

- [ ] **Step 4: Write the routes**

Create `src/eo_visual_retrieval/app/main.py`:

```python
"""HTTP routing for the comparison surface.

This is the only module here that knows about HTTP. All ranking decisions live
in catalog.py so they can be tested without a server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from eo_visual_retrieval.app.catalog import Catalog
from eo_visual_retrieval.app.thumbnails import thumbnail_jpeg
from eo_visual_retrieval.app.uploads import MAX_UPLOAD_BYTES, decode_upload

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
THUMBNAIL_PIXELS = 96


def create_app(catalog: Catalog, *, k: int = 5) -> FastAPI:
    """Build the application around one already-validated catalog."""

    if k < 1:
        raise ValueError("k must be positive")
    app = FastAPI(title="EO visual retrieval")

    def render(
        request: Request,
        rankings: list[Any],
        *,
        query_id: str | None,
        is_upload: bool,
    ) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request=request,
            name="compare.html",
            context={
                "rankings": rankings,
                "query_ids": catalog.query_ids,
                "item_id": query_id,
                "query_id": query_id,
                "is_upload": is_upload,
                "upload_available": catalog.upload_available,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        first = catalog.query_ids[0]
        return render(request, catalog.rank_item(first, k=k), query_id=first, is_upload=False)

    @app.get("/compare", response_class=HTMLResponse)
    def compare(request: Request, item_id: str) -> HTMLResponse:
        try:
            rankings = catalog.rank_item(item_id, k=k)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return render(request, rankings, query_id=item_id, is_upload=False)

    @app.post("/compare/upload", response_class=HTMLResponse)
    async def compare_upload(
        request: Request, image: UploadFile = File(...)
    ) -> HTMLResponse:
        data = await image.read(MAX_UPLOAD_BYTES + 1)
        try:
            pixels = decode_upload(data, image_size=catalog.image_size)
            ranking = catalog.rank_uploaded(pixels, k=k)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return render(request, [ranking], query_id=None, is_upload=True)

    @app.get("/thumbnail")
    def thumbnail(item_id: str) -> Response:
        try:
            path = catalog.image_path(item_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return Response(
            content=thumbnail_jpeg(path, size=THUMBNAIL_PIXELS), media_type="image/jpeg"
        )

    return app
```

- [ ] **Step 5: Add the `serve` subcommand**

In `src/eo_visual_retrieval/cli.py`, add this handler beside the other `_`-prefixed handlers:

```python
def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from eo_visual_retrieval.app.catalog import Catalog
    from eo_visual_retrieval.app.main import create_app

    catalog = Catalog.load(
        manifest=args.manifest,
        image_root=args.image_root,
        stores=args.store,
        projection=args.projection,
    )
    print(
        json.dumps(
            {
                "stores": [str(path) for path in args.store],
                "queries": len(catalog.query_ids),
                "upload_available": catalog.upload_available,
                "url": f"http://{args.host}:{args.port}",
            },
            indent=2,
        )
    )
    uvicorn.run(create_app(catalog, k=args.k), host=args.host, port=args.port)
```

Then, immediately before `return parser` at line 684, add:

```python
    serve = commands.add_parser(
        "serve",
        help="serve a representation-comparison view over prepared embedding stores",
    )
    serve.add_argument("--manifest", type=Path, required=True)
    serve.add_argument("--image-root", type=Path, required=True)
    serve.add_argument(
        "--store",
        type=Path,
        action="append",
        required=True,
        help="an embedding store to compare; repeat to add representations, which "
        "are shown in the order given",
    )
    serve.add_argument(
        "--projection",
        type=Path,
        help="PCA basis from embed-pca --projection-output; without it the upload "
        "path is disabled",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--k", type=int, default=5)
    serve.set_defaults(handler=_serve)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest tests/test_app_routes.py -q`
Expected: PASS, 8 tests

- [ ] **Step 7: Run the full gates**

```bash
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m ruff check .
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m mypy
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest -q --cov=eo_visual_retrieval --cov-report=term --cov-fail-under=75
```

Expected: all clean.

- [ ] **Step 8: Commit**

```bash
git add src/eo_visual_retrieval/app/main.py src/eo_visual_retrieval/app/templates src/eo_visual_retrieval/cli.py tests/test_app_routes.py
git commit -m "$(cat <<'EOF'
Serve the comparison view behind eovr serve

Routing is the only part that knows about HTTP; every ranking decision stays in
the catalog. Thumbnail paths come from the catalog rather than the request, so a
traversal attempt resolves to 404 instead of touching disk.

An uploaded image carries no label, so its results are shown uncoloured and the
page says grey means unknown rather than wrong.

A test asserts the served modules import no model framework, which is what keeps
the deployable image small.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Run it against real data and document it

**Files:**
- Modify: `docs/pipeline-and-cli.md`, `docs/architecture.md`, `docs/development.md`, `docs/project-context.md`, `README.md`, `docs/validation.md`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing later tasks depend on. This is the final task.

- [ ] **Step 1: Generate the PCA projection the published store never saved**

The committed PCA store predates `--projection-output`, so the basis must be written out. The fit is deterministic given the same manifest, images, seed, and component count, so this reproduces the basis that produced the published store. Write the store to a throwaway path and keep only the projection:

```bash
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m eo_visual_retrieval.cli embed-pca --manifest data/eurosat-v1/manifest.jsonl --image-root data/eurosat-v1/images --output outputs/pca-64-regenerated.npz --projection-output artifacts/eurosat-v1-pca-64-projection.npz --components 64 --image-size 64 --seed 42
```

Then confirm the regenerated vectors match the published store, so the basis is the right one:

```bash
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -c "import numpy as np; from eo_visual_retrieval.embeddings.store import EmbeddingStore as S; from pathlib import Path; a=S.load(Path('artifacts/eurosat-v1-pca-64.npz')); b=S.load(Path('outputs/pca-64-regenerated.npz')); print('ids match:', a.ids==b.ids); print('max abs diff:', float(np.abs(a.vectors-b.vectors).max()))"
```

Expected: `ids match: True` and a maximum absolute difference below `1e-4`. If it is larger, stop — the basis does not correspond to the published store.

- [ ] **Step 2: Serve the real corpus and exercise it**

```bash
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m eo_visual_retrieval.cli serve --manifest data/eurosat-v1/manifest.jsonl --image-root data/eurosat-v1/images --store artifacts/eurosat-v1-pca-64.npz --store artifacts/eurosat-v1-dinov2-vits14.npz --store artifacts/eurosat-v1-ssl4eo-s12-rgb-moco-resnet50.npz --store artifacts/eurosat-v1-ssl4eo-s12-moco-resnet50.npz --projection artifacts/eurosat-v1-pca-64-projection.npz --port 8000
```

Expected startup JSON: `"queries": 400` and `"upload_available": true`.

With the server running, in a second shell confirm the routes answer and record the observed values:

```bash
curl -s -o NUL -w "home %{http_code} %{time_total}s\n" http://127.0.0.1:8000/
curl -s -o NUL -w "compare %{http_code}\n" "http://127.0.0.1:8000/compare?item_id=River/River_1000.tif"
curl -s -o NUL -w "thumb %{http_code} %{size_download}B\n" "http://127.0.0.1:8000/thumbnail?item_id=River/River_1.tif"
curl -s -o NUL -w "unknown %{http_code}\n" "http://127.0.0.1:8000/compare?item_id=absent.tif"
```

Expected: `200` for the first three and `404` for the last. If `River/River_1000.tif` is not in the manifest, pick any ID from the startup listing instead and record which one was used.

Stop the server when finished.

- [ ] **Step 3: Document the command**

In `docs/pipeline-and-cli.md`, add a section immediately before `## Reproducibility checklist`:

```markdown
## Serve the comparison surface

```powershell
eovr serve `
  --manifest data/eurosat-v1/manifest.jsonl `
  --image-root data/eurosat-v1/images `
  --store artifacts/eurosat-v1-pca-64.npz `
  --store artifacts/eurosat-v1-dinov2-vits14.npz `
  --store artifacts/eurosat-v1-ssl4eo-s12-rgb-moco-resnet50.npz `
  --store artifacts/eurosat-v1-ssl4eo-s12-moco-resnet50.npz `
  --projection artifacts/eurosat-v1-pca-64-projection.npz
```

`--store` is repeatable, and representations appear in the order given. Placing the SSL4EO 13-band
store beside its RGB variant shows the band ablation as rankings rather than as a table.

The server loads only precomputed vectors, so it imports no model framework. Ranking is a
matrix-vector product; an uploaded image is embedded with the persisted PCA basis, the one
representation this project fits itself. Without `--projection` the upload path is disabled and the
page says so.

The catalog refuses to start when the supplied stores disagree on their manifest hash or item
ordering, because a comparison across different corpora would be meaningless while looking correct.

Uploads carry no label, so their results are shown without relevance colouring and without a
per-query metric. Grey means unknown, not wrong.
```

- [ ] **Step 4: Record the component in the architecture table**

In `docs/architecture.md`, add these rows to the component table immediately after the
`Result-grid renderer` row:

```markdown
| Served catalog | Rank one query with several representations and report their provenance | `app/catalog.py` |
| Comparison surface | Route, render, and accept uploads over the served catalog | `app/main.py` |
```

Then, in `docs/architecture.md`, replace this limitation bullet:

```markdown
- There is no product API, serving database, job runner, or interactive retrieval viewer.
```

with:

```markdown
- The interactive viewer serves precomputed vectors only. There is still no serving database or
  job runner, and uploads are embedded with PCA alone, because the other representations would
  require a model framework in the served process.
```

- [ ] **Step 5: Record the dependency group**

In `docs/development.md`, add to the optional-group list, immediately after the `search` bullet:

```markdown
- `app`: FastAPI, Uvicorn, Jinja2, and python-multipart for the served comparison surface;
```

- [ ] **Step 6: Record executed evidence**

In `docs/validation.md`, insert this section immediately before the line
`## SSL4EO-S12 band ablation on EuroSAT v1 — 2026-09-03`, filling in the values observed in Step 2:

```markdown
## Comparison surface smoke — 2026-09-03

Executed against the local EuroSAT v1 corpus with four representations loaded.

| Gate | Executed evidence | Status |
|---|---|---|
| PCA basis | Regenerated deterministically; ordered IDs matched the published store and the maximum absolute vector difference was <OBSERVED> | Passed |
| Startup | 400 query items loaded, upload available, four stores accepted | Passed |
| Comparison route | HTTP 200 for a corpus query across all four representations | Passed |
| Thumbnail route | HTTP 200, JPEG, <OBSERVED> bytes | Passed |
| Unknown item | HTTP 404 | Passed |
| No model framework | Test asserts torch, torchvision, terratorch, and sklearn stay unimported by the served modules | Passed |

This is an execution smoke on one machine. It establishes that the surface serves the recorded
corpus and refuses malformed input; it measures no latency under load, has not been deployed, and
produces no retrieval evidence. All rankings come from previously published embedding stores.
```

- [ ] **Step 7: Point the roadmap at what remains**

In `docs/project-context.md`, replace the `### Milestone 5: usable project surface` body with:

```markdown
Exposed as `eovr serve`: a representation-comparison view over the prepared EuroSAT v1 stores,
where one query is ranked by every supplied model with its provenance, and an uploaded image is
ranked through the persisted PCA basis. See [Pipeline and CLI](pipeline-and-cli.md).

What remains is deployment and breadth, not capability: the surface has not been published
anywhere, uploads work for PCA only because the other representations would put a model framework
in the served process, and it serves EuroSAT v1 alone. A public deployment needs no paid
infrastructure — the served payload is roughly 35 MB with no GPU and no model framework — so it
remains a reversible step rather than a cost decision.
```

- [ ] **Step 8: Run the full gates and confirm nothing published moved**

```bash
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m ruff check .
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m mypy
C:\Users\nikos\.venvs\eovr\Scripts\python.exe -m pytest -q --cov=eo_visual_retrieval --cov-report=term --cov-fail-under=75
git status --short docs/results/ docs/assets/ src/eo_visual_retrieval/evaluation.py src/eo_visual_retrieval/retrieval.py
```

Expected: gates clean, and the `git status` prints nothing.

- [ ] **Step 9: Commit**

```bash
git add docs README.md
git commit -m "$(cat <<'EOF'
Document the comparison surface and record its smoke run

Adds the serve command to the pipeline guide, the two new components to the
architecture table, the app dependency group, and an executed smoke to the
validation record.

Milestone 5's remaining work is now deployment and breadth rather than
capability, and the roadmap says so: nothing is published anywhere, uploads work
for PCA only, and the corpus is EuroSAT v1.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task. Shape and the `app` extra to Task 1 Step 1; the four components to Tasks 1-4; the comparability invariant to Task 1 Steps 2 and 4; the comparison view, colour language, provenance, and the upload asymmetry to Task 4 Steps 3-4; untrusted input to Task 3; failure behaviour to Tasks 1, 3, and 4; documentation and the executed smoke to Task 5. Spec acceptance criteria 1-7 map to Task 5 Step 2, Task 5 Step 2, Task 1 Step 2, Task 3 Step 1, Task 1 Step 2, Task 5 Step 8, and Task 5 Step 8 respectively.

**Deployment is deliberately not built.** The spec calls for a host-agnostic container definition, but publishing was explicitly out of scope for now and a Containerfile with nothing to deploy to would be untested decoration. Task 5 Step 7 records that a deployment needs no paid infrastructure so the decision stays open. If you want the Containerfile in this pass, say so and it becomes a sixth task.

**Placeholder scan.** The two `<OBSERVED>` markers in Task 5 Step 6 are deliberate: they are values a run produces, and the project's evidence policy forbids writing a number before a run produces it. Step 2 names exactly which commands produce them. No other placeholders.

**Type consistency.** `Catalog.load` takes `manifest`, `image_root`, `stores`, `projection` as keyword arguments in Task 1's definition, its tests, Task 4's test fixture, and the CLI handler. `rank_item(item_id, *, k)` returns `list[ModelRanking]`; `rank_uploaded(pixels, *, k)` returns a single `ModelRanking`, which Task 4 wraps in a list before rendering. `RankedResult.relevant` is `bool | None` and the template branches on `is none` first. `thumbnail_jpeg(path, *, size, quality)` and `decode_upload(data, *, image_size)` match their call sites. `create_app(catalog, *, k)` matches both the test fixture and the CLI.

**Risk noted.** Task 5 Step 1 regenerates a PCA basis for a store committed before the flag existed. Step 1 verifies the regenerated vectors match the published store before the basis is used, and stops if they do not.
