"""Routes, and the guarantee that the served process carries no model framework."""

from __future__ import annotations

import asyncio
import hashlib
import io
import sys
from dataclasses import replace
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
from starlette.types import Message, Receive, Scope, Send  # noqa: E402

from eo_visual_retrieval.app.main import ContentLengthLimitMiddleware, create_app  # noqa: E402

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
    for position, record in enumerate(records):
        destination = tmp_path / record.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 60 + position * 30, dtype=np.uint8)
        ).save(destination, format="TIFF")
        record.metadata["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()

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
    projection = PcaProjection(
        mean=np.zeros(FEATURES, dtype=np.float32),
        components=np.eye(2, FEATURES, dtype=np.float32) + 0.5,
        image_size=IMAGE_SIZE,
        seed=42,
    )
    projection.save(projection_path)
    replace(
        store,
        vectors=projection.embed_images([tmp_path / record.path for record in records]),
    ).save(store_path)

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
    assert 'role="alert"' in response.text
    assert "Return to the explorer" in response.text


def test_health_and_missing_form_field(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    response = client.get("/compare")
    assert response.status_code == 422
    assert 'role="alert"' in response.text


def test_query_picker_and_result_identity_are_labelled(client: TestClient) -> None:
    page = client.get("/").text
    assert '<optgroup label="forest">' in page
    assert 'alt="Rank 1: forest"' in page
    assert "forest/a.tif" in page
    assert "Same class" in page
    assert "Model &amp; data provenance" in page
    assert 'name="viewport"' in page


@pytest.mark.parametrize("declared", [None, b"1", b"bad", b"-1"])
def test_actual_body_is_bounded_before_parser(declared: bytes | None) -> None:
    """Chunked or falsely small declared bodies must never reach the parser."""
    called = False
    sent: list[Message] = []
    messages: list[Message] = [
        {"type": "http.request", "body": b"1234", "more_body": True},
        {"type": "http.request", "body": b"5678", "more_body": False},
    ]
    chunks = iter(messages)

    async def parser(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal called
        called = True

    async def receive() -> Message:
        return next(chunks)

    async def send(message: Message) -> None:
        sent.append(message)

    headers = [] if declared is None else [(b"content-length", declared)]
    asyncio.run(ContentLengthLimitMiddleware(parser, max_bytes=6)(
        {"type": "http", "headers": headers}, receive, send
    ))
    assert not called
    assert sent[0]["status"] == (400 if declared in (b"bad", b"-1") else 413)


def test_bounded_chunked_body_is_replayed_and_disconnect_is_safe() -> None:
    consumed: list[Message] = []

    async def run(disconnected: bool) -> None:
        chunks: list[Message] = [
            {"type": "http.request", "body": b"12", "more_body": True},
            {"type": "http.disconnect"} if disconnected else
            {"type": "http.request", "body": b"34", "more_body": False},
        ]

        async def receive() -> Message:
            return chunks.pop(0)

        async def parser(scope: Scope, receive: Receive, send: Send) -> None:
            consumed.append(await receive())

        async def send(message: Message) -> None:
            pass

        await ContentLengthLimitMiddleware(parser, max_bytes=4)(
            {"type": "http", "headers": []}, receive, send
        )

    asyncio.run(run(disconnected=True))
    assert consumed == []
    asyncio.run(run(disconnected=False))
    assert consumed == [{"type": "http.request", "body": b"1234", "more_body": False}]


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


def test_findings_page_is_absent_unless_evidence_is_supplied(tmp_path: Path) -> None:
    """A deployment without the reports serves no findings, rather than an empty page."""

    client = TestClient(create_app(_catalog(tmp_path), k=2))

    assert client.get("/findings").status_code == 404
    assert "/findings" not in client.get("/").text


def test_findings_page_renders_from_the_supplied_payload(tmp_path: Path) -> None:
    from eo_visual_retrieval.app.findings import build_payload

    results = Path(__file__).resolve().parents[1] / "docs" / "results"
    payload = build_payload(results)
    client = TestClient(create_app(_catalog(tmp_path), k=2, findings=payload))

    page = client.get("/findings")
    assert page.status_code == 200
    body = page.text
    # The charts are drawn from an embedded payload, not from markup.
    assert 'id="findings-data"' in body
    assert "/static/findings.js" in body
    for series in payload["series"]:
        assert series["key"] in body
    # The explorer links to it once it exists.
    assert "/findings" in client.get("/").text
