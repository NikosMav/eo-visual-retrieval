"""Ranking contracts with hand-computable vectors, independent of model downloads."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from PIL import Image

from eo_visual_retrieval.app.catalog import Catalog
from eo_visual_retrieval.app.evidence import Evidence
from eo_visual_retrieval.app.search import create_search_app, decode_search_image
from eo_visual_retrieval.cli import build_parser
from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.hashing import file_sha256
from eo_visual_retrieval.manifests import read_jsonl, write_jsonl
from eo_visual_retrieval.models import ImageRecord
from eo_visual_retrieval.multimodal import MultimodalSearch
from eo_visual_retrieval.search_plan import (
    SearchFilters,
    SearchPlan,
    matches_filters,
    plan_query,
    scene_metadata,
)


class FakeEncoder:
    metadata: dict[str, Any] = {"backend": "fixture", "model": "fixture", "dimension": 2}
    calls = 0

    def encode_text(self, text: str) -> NDArray[np.float32]:
        self.calls += 1
        return np.asarray([1, 0], dtype=np.float32)

    def encode_image(self, image: Image.Image) -> NDArray[np.float32]:
        self.calls += 1
        return np.asarray([0, 1], dtype=np.float32)


@pytest.fixture
def engine(tmp_path: Path) -> MultimodalSearch:
    records = []
    for i, item_id in enumerate(("text", "image", "both", "missing", "query")):
        Image.new("RGB", (12, 8), (20 + i, 40, 60)).save(tmp_path / f"{item_id}.png")
        metadata = (
            {}
            if item_id == "missing"
            else {
                "centroid_lonlat": [23.7, 37.98],
                "datetime": "2026-09-01T10:00:00Z",
                "eo_cloud_cover": 4 if item_id != "text" else 80,
                "collection": "sentinel-2-l2a",
                "private_token": "never expose me",
            }
        )
        records.append(
            ImageRecord(
                item_id,
                f"{item_id}.png",
                "query" if item_id == "query" else "index",
                metadata=metadata,
            )
        )
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(records, manifest)
    vectors = {"text": [1, 0], "image": [0, 1], "both": [1, 1], "missing": [1, 0], "query": [-1, 0]}
    records.sort(key=lambda r: r.item_id)
    store = EmbeddingStore(
        ids=tuple(r.item_id for r in records),
        labels=(None,) * 5,
        splits=tuple(r.split for r in records),
        vectors=np.asarray([vectors[r.item_id] for r in records], dtype=np.float32),
        metadata={**FakeEncoder.metadata, "manifest_sha256": file_sha256(manifest)},
    )
    store.save(tmp_path / "store.npz")
    return MultimodalSearch.load(
        manifest=manifest,
        embeddings=tmp_path / "store.npz",
        image_root=tmp_path,
        encoder=FakeEncoder(),
    )


def test_text_image_and_hybrid_rankings(engine: MultimodalSearch) -> None:
    image = Image.new("RGB", (8, 8))
    text = engine.search(SearchPlan("urban"), k=3)
    visual = engine.search(SearchPlan(""), image=image, k=3)
    hybrid = engine.search(SearchPlan("urban"), image=image, k=3)
    assert [r["item_id"] for r in text["results"]] == ["missing", "text", "both"]
    assert visual["results"][0]["item_id"] == "image"
    assert hybrid["results"][0]["item_id"] == "both"
    assert hybrid["results"][0]["score"] == pytest.approx(2**-0.5)
    for row in hybrid["results"]:
        assert row["score"] == pytest.approx(0.5 * row["text_score"] + 0.5 * row["image_score"])
    assert "query" not in [r["item_id"] for r in hybrid["results"]]
    assert text["results"][0]["image_score"] is None
    assert visual["results"][0]["text_score"] is None


def test_explanation_matches_ranking_and_filter_counts(engine: MultimodalSearch) -> None:
    result = engine.search(SearchPlan("urban", SearchFilters(max_cloud_cover=10)),
                           image=Image.new("RGB", (8, 8)), k=1)
    assert result["diagnostics"]["filter_counts"] == {
        "max_cloud_cover": {"pass": 2, "fail": 1, "missing": 1}}
    assert result["diagnostics"]["excluded_by_filters"] == 2
    assert result["diagnostics"]["elapsed_ms"] >= 0
    row = result["results"][0]
    assert row["rank"] == 1
    assert row["explanation"]["text_rank"] == 1
    assert row["explanation"]["image_rank"] == 2
    assert row["explanation"]["filter_checks"] == {"max_cloud_cover": "pass"}
    assert row["score"] == pytest.approx(row["explanation"]["text_contribution"]
                                         + row["explanation"]["image_contribution"])
    selected = engine.search(SearchPlan("urban"), item_id="both")
    assert selected["diagnostics"]["excluded_example"] == 1
    assert selected["provenance"]["manifest_sha256"] == engine.store.metadata["manifest_sha256"]


def test_product_pages_and_mounted_comparison(engine: MultimodalSearch, tmp_path: Path) -> None:
    catalog = Catalog(read_jsonl(tmp_path / "manifest.jsonl"), [engine.store], image_root=tmp_path)
    client = TestClient(create_search_app(engine, comparison=catalog))
    for url in ("/", "/findings", "/research", "/models/"):
        response = client.get(url)
        assert response.status_code == 200
        assert 'aria-label="Product"' in response.text
    compare = client.get("/models/").text
    assert 'action="/models/compare"' in compare
    assert 'src="/models/thumbnail?' in compare
    assert client.get("/models/compare", params={"item_id": "query"}).status_code == 200
    assert client.get("/models/thumbnail", params={"item_id": "both"}).status_code == 200
    assert client.get("/models/static/explorer.css").status_code == 200
    assert "No evidence reports" in client.get("/findings").text
    assert client.get("/evidence/private.json").status_code == 404
    without = TestClient(create_search_app(engine))
    assert "not configured" in without.get("/models/").text


def test_evidence_snapshot_and_public_allowlist(engine: MultimodalSearch, tmp_path: Path) -> None:
    public = tmp_path / "multimodal-v1-smoke.json"
    public.write_text('{"quality_measured": false}', encoding="utf-8")
    (tmp_path / "secrets.json").write_text('{"token": "private"}', encoding="utf-8")
    client = TestClient(create_search_app(engine, results_dir=tmp_path))
    public.write_text("{}", encoding="utf-8")
    assert client.get("/evidence/multimodal-v1-smoke.json").json() == {"quality_measured": False}
    assert client.get("/evidence/secrets.json").status_code == 404
    sources = client.get("/api/evidence").json()["sources"]
    assert len(sources) == 1 and len(sources[0]["sha256"]) == 64
    with pytest.raises(ValueError, match="directory"):
        Evidence(tmp_path / "absent")
    public.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        Evidence(tmp_path)


def test_committed_evidence_renders(engine: MultimodalSearch) -> None:
    reports = Path(__file__).resolve().parents[1] / "docs" / "results"
    client = TestClient(create_search_app(engine, results_dir=reports))
    page = client.get("/findings")
    assert page.status_code == 200
    assert "41.7%" in page.text and "80.6%" in page.text
    assert "400" in page.text and "nDCG@10" in page.text
    assert client.get("/findings/analysis").status_code == 200
    assert "private" not in client.get("/api/evidence").text


def test_hybrid_endpoints_and_cancellation(engine: MultimodalSearch) -> None:
    image = Image.new("RGB", (8, 8))
    assert (
        engine.search(SearchPlan("urban"), image=image, text_weight=1)["results"][0]["item_id"]
        == "missing"
    )
    assert (
        engine.search(SearchPlan("urban"), image=image, text_weight=0)["results"][0]["item_id"]
        == "image"
    )
    cancelled = engine.search(SearchPlan("urban"), item_id="query")
    assert all(r["score"] == 0 for r in cancelled["results"])
    assert [r["item_id"] for r in cancelled["results"]] == ["both", "image", "missing", "text"]


def test_prefilters_precede_top_k_and_unknown_is_excluded(engine: MultimodalSearch) -> None:
    result = engine.search(SearchPlan("urban", SearchFilters(max_cloud_cover=10)), k=1)
    assert result["candidate_count"] == 2
    assert result["results"][0]["item_id"] == "both"
    assert "private_token" not in json.dumps(result)
    assert "never expose" not in json.dumps(result)
    empty = engine.search(SearchPlan("urban", SearchFilters(start_date=date(2027, 1, 1))))
    assert empty["candidate_count"] == 0 and empty["results"] == []
    assert engine.encoder.calls == 1  # type: ignore[attr-defined]
    assert "Missing metadata" in empty["message"]


def test_example_exclusion_and_large_k(engine: MultimodalSearch) -> None:
    response = engine.search(SearchPlan(""), item_id="image", k=100)
    assert len(response["results"]) == 3
    assert all(r["item_id"] != "image" for r in response["results"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"k": 101},
        {"text_weight": float("nan")},
        {"text_weight": 1.1},
        {"item_id": "unknown"},
    ],
)
def test_invalid_queries(engine: MultimodalSearch, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        engine.search(SearchPlan("urban"), **kwargs)


def test_empty_or_conflicting_query(engine: MultimodalSearch) -> None:
    with pytest.raises(ValueError, match="provide"):
        engine.search(SearchPlan(" "))
    with pytest.raises(ValueError, match="not both"):
        engine.search(SearchPlan("urban"), item_id="image", image=Image.new("RGB", (1, 1)))


def test_query_vector_validation(engine: MultimodalSearch, monkeypatch: pytest.MonkeyPatch) -> None:
    for vector in ([0, 0], [float("nan"), 0], [1, 0, 0]):
        monkeypatch.setattr(engine.encoder, "encode_text", lambda text, v=vector: np.array(v))
        with pytest.raises(ValueError):
            engine.search(SearchPlan("urban"))


def test_store_and_image_binding(engine: MultimodalSearch, tmp_path: Path) -> None:
    records = list(engine._records.values())
    for store in (
        replace(engine.store, metadata={**engine.store.metadata, "model": "other"}),
        replace(engine.store, vectors=np.ones((5, 3), dtype=np.float32)),
    ):
        with pytest.raises(ValueError):
            MultimodalSearch(records, store, FakeEncoder(), image_root=tmp_path)
    with pytest.raises(ValueError, match="order"):
        MultimodalSearch(records[::-1], engine.store, FakeEncoder(), image_root=tmp_path)
    records[0] = replace(records[0], metadata={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="checksum"):
        MultimodalSearch(records, engine.store, FakeEncoder(), image_root=tmp_path)
    (tmp_path / "manifest.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        MultimodalSearch.load(
            manifest=tmp_path / "manifest.jsonl",
            embeddings=tmp_path / "store.npz",
            image_root=tmp_path,
            encoder=FakeEncoder(),
        )


@pytest.mark.parametrize("path", ["../outside.png", "C:/Windows/win.ini", "/etc/passwd", "C:foo"])
def test_path_containment(engine: MultimodalSearch, tmp_path: Path, path: str) -> None:
    records = list(engine._records.values())
    records[0] = replace(records[0], path=path)
    with pytest.raises(ValueError, match="path|outside"):
        MultimodalSearch(records, engine.store, FakeEncoder(), image_root=tmp_path)


def test_prompt_defaults_are_visible_and_overridable() -> None:
    text = "Sentinel imagery showing recent urban expansion near Athens with low cloud coverage."
    plan = plan_query(text, today=date(2026, 9, 5))
    assert plan.filters == SearchFilters(
        (23.4, 37.7, 24.1, 38.2), date(2026, 6, 7), date(2026, 9, 5), 10, "sentinel-2-l2a"
    )
    assert "Single-scene" in " ".join(plan.notes)
    assert plan_query(text, interpret=False).filters == SearchFilters()
    explicit = SearchFilters(
        start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), max_cloud_cover=0
    )
    override = plan_query(text, overrides=explicit)
    assert override.filters.start_date == date(2024, 1, 1)
    assert override.filters.max_cloud_cover == 0
    assert plan_query("not near Athens, recent imagery").filters == SearchFilters()
    with pytest.raises(ValueError):
        plan_query("x" * 1001)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bbox": (180, 0, -180, 5)},
        {"bbox": (0, 0, float("nan"), 5)},
        {"bbox": (0, 0, 1)},
        {"max_cloud_cover": -1},
        {"max_cloud_cover": float("nan")},
        {"collection": "https://example.com?token=secret"},
        {"start_date": date(2026, 1, 2), "end_date": date(2026, 1, 1)},
    ],
)
def test_invalid_filters(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SearchFilters(**kwargs)


def test_metadata_validation_and_date_boundaries() -> None:
    record = ImageRecord(
        "a",
        "a.png",
        "index",
        metadata={
            "datetime": "2026-09-02T01:00:00+03:00",
            "eo_cloud_cover": 0,
            "bbox": [23, 37, 24, 38],
            "bbox_crs": "EPSG:4326",
            "collection": "sentinel-2-l2a",
        },
    )
    metadata = scene_metadata(record)
    assert metadata["date"] == "2026-09-01"
    assert metadata["centroid_lonlat"] == (23.5, 37.5)
    assert matches_filters(
        metadata,
        SearchFilters(
            (23.5, 37.5, 24, 38), date(2026, 9, 1), date(2026, 9, 1), 0, "sentinel-2-l2a"
        ),
    )
    for filters in (
        SearchFilters(end_date=date(2026, 8, 31)),
        SearchFilters(collection="other"),
        SearchFilters(bbox=(0, 0, 1, 1)),
    ):
        assert not matches_filters(metadata, filters)
    for raw in (
        {"datetime": "bad", "eo_cloud_cover": float("nan")},
        {"datetime": "2026-09-01", "centroid_lonlat": [300, 91]},
        {"centroid_lonlat": ["bad", 1], "collection": "https://private"},
        {"bbox": ["bad", 1, 2, 3], "bbox_crs": "EPSG:4326"},
    ):
        row = scene_metadata(replace(record, metadata=raw))
        assert row["date"] is None and row["centroid_lonlat"] is None
        assert not matches_filters(row, SearchFilters(bbox=(0, 0, 1, 1)))
        assert not matches_filters(row, SearchFilters(start_date=date(2020, 1, 1)))


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (20, 10)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_api_text_upload_hybrid_and_plan(engine: MultimodalSearch) -> None:
    with TestClient(create_search_app(engine)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/api/corpus").json()["index_items"] == 4
        assert (
            client.get("/thumbnail", params={"item_id": "image"}).headers["content-type"]
            == "image/jpeg"
        )
        assert client.get("/thumbnail", params={"item_id": "../outside"}).status_code == 404
        assert client.post("/api/plan", json={"text": "recent"}).json()["filters"]["start_date"]
        for text, image, mode in (
            ("urban", False, "text"),
            ("", True, "image"),
            ("urban", True, "hybrid"),
        ):
            files = {"image": ("test.png", _png(), "image/png")} if image else None
            response = client.post(
                "/api/search", data={"query": json.dumps({"text": text})}, files=files
            )
            assert response.status_code == 200, response.text
            assert response.json()["mode"] == mode
        response = client.post("/api/search", data={"query": json.dumps({"item_id": "image"})})
        assert response.status_code == 200


def test_api_errors_and_upload_bounds(
    engine: MultimodalSearch, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eo_visual_retrieval.app import search

    with TestClient(create_search_app(engine)) as client:
        for payload in ({"text_weight": 2}, {"unknown": "field"}, {"start_date": "wrong"}):
            assert client.post("/api/plan", json=payload).status_code == 422
            assert (
                client.post("/api/search", data={"query": json.dumps(payload)}).status_code == 422
            )
        assert client.post("/api/plan", json={"bbox": [3, 2, 1, 0]}).status_code == 400
        assert client.post("/api/search", data={"query": "{}"}).status_code == 400
        assert client.post("/api/search", data={"query": "no json"}).status_code == 422
        assert client.post("/api/search", data={}).status_code == 400
        assert client.post("/api/search", content=b"x" * (8 * 1024 * 1024 + 1)).status_code == 413
        response = client.post(
            "/api/search",
            data={"query": '{"text":"urban"}'},
            files={"image": ("bad.png", b"garbage", "image/png")},
        )
        assert response.status_code == 400
    decoded = decode_search_image(_png())
    assert decoded.size == (20, 10)  # Preserve aspect ratio until model preprocessing.
    decoded.close()
    for data in (b"", b"bad", b"x" * (8 * 1024 * 1024 + 1)):
        with pytest.raises(ValueError):
            decode_search_image(data)
    monkeypatch.setattr(search, "MAX_UPLOAD_PIXELS", 5)
    with pytest.raises(ValueError, match="megapixel"):
        decode_search_image(_png())


def test_cli_plan_requires_no_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = build_parser().parse_args(
        [
            "search",
            "--manifest",
            str(tmp_path / "absent.jsonl"),
            "--image-root",
            str(tmp_path),
            "--embeddings",
            str(tmp_path / "absent.npz"),
            "--text",
            "recent",
            "--plan-only",
        ]
    )
    args.handler(args)
    assert json.loads(capsys.readouterr().out)["filters"]["start_date"]
