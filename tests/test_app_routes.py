"""Routes, and the guarantee that the served process carries no model framework."""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eo_visual_retrieval.app.catalog import Catalog
from eo_visual_retrieval.app.uploads import MAX_UPLOAD_BYTES
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
        ImageRecord(item_id=i, path=i, split=s, label=lab, metadata={"sha256": str(n) * 64})
        for n, (i, s, lab) in enumerate(rows)
    ]
    for record in records:
        destination = tmp_path / record.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 60, dtype=np.uint8)
        ).save(destination, format="TIFF")

    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(records, manifest)
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    store = EmbeddingStore(
        ids=tuple(r.item_id for r in records),
        vectors=np.asarray(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.99, 0.01]], dtype=np.float32
        ),
        labels=tuple(r.label for r in records),
        splits=tuple(r.split for r in records),
        metadata={
            "backend": "pca",
            "manifest_sha256": manifest_digest,
            "image_size": IMAGE_SIZE,
        },
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
    # The legend must name every colour the tiles can actually render, including
    # grey, and state that row order is not a ranking.
    assert "grey" in response.text.lower()
    assert "not ranked by quality" in response.text.lower()
    # Serving EuroSAT imagery is conditioned on carrying this attribution.
    assert "eurosat" in response.text.lower()
    assert "mit" in response.text.lower()


def test_comparison_names_the_model_and_its_provenance(
    client: TestClient, tmp_path: Path
) -> None:
    manifest_digest = hashlib.sha256((tmp_path / "manifest.jsonl").read_bytes()).hexdigest()

    response = client.get("/compare", params={"item_id": "forest/q.tif"})

    assert response.status_code == 200
    assert "pca" in response.text
    assert manifest_digest in response.text


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


def test_thumbnail_of_a_missing_source_file_is_not_found(
    client: TestClient, tmp_path: Path
) -> None:
    """catalog.image_path resolving the ID does not mean the file still exists.

    thumbnail_jpeg raises ValueError for a missing source and can raise OSError
    for a corrupt one; the route must turn both into 404, not a 500.
    """
    (tmp_path / "forest" / "a.tif").unlink()

    response = client.get("/thumbnail", params={"item_id": "forest/a.tif"})

    assert response.status_code == 404


def test_create_app_refuses_k_larger_than_the_index(tmp_path: Path) -> None:
    """k is validated once at startup so a small corpus fails fast, not per request."""
    catalog = _catalog(tmp_path)

    with pytest.raises(ValueError, match="exceeds the index size"):
        create_app(catalog, k=100)


def test_oversized_declared_upload_is_rejected_before_form_parsing(client: TestClient) -> None:
    """A declared Content-Length over the cap is refused before Starlette spools it."""
    response = client.post(
        "/compare/upload",
        headers={"content-length": str(MAX_UPLOAD_BYTES + 1)},
        content=b"short body; only the declared header is checked here",
    )

    assert response.status_code == 413


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
    """The deployable image stays small only if nothing pulls in a framework.

    This must run in a fresh interpreter. Other tests in this suite import
    scikit-learn and torch at module scope, so checking sys.modules in the shared
    pytest process would measure test ordering rather than what the app imports.
    """
    import subprocess

    probe = (
        "import sys;"
        "import eo_visual_retrieval.app.catalog;"
        "import eo_visual_retrieval.app.main;"
        "import eo_visual_retrieval.app.thumbnails;"
        "import eo_visual_retrieval.app.uploads;"
        "leaked=[m for m in ('torch','torchvision','terratorch','sklearn') "
        "if m in sys.modules];"
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "", (
        f"the served surface imported a model framework: {result.stdout.strip()}"
    )
