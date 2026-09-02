"""End-to-end CLI wiring: argument contracts, artifacts, and error formatting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eo_visual_retrieval.cli import build_parser, main
from eo_visual_retrieval.embeddings.store import EmbeddingStore

pytest.importorskip("sklearn")

IMAGE_SIZE = 8


def _run(capsys: pytest.CaptureFixture[str], *arguments: str) -> Any:
    """Parse real CLI arguments, run the handler, and return its JSON output."""

    args = build_parser().parse_args(list(arguments))
    args.handler(args)
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def images(tmp_path: Path) -> Path:
    from PIL import Image

    root = tmp_path / "images"
    generator = np.random.default_rng(11)
    for label, base in (("forest", 40), ("water", 200)):
        (root / label).mkdir(parents=True)
        for position in range(6):
            noise = generator.integers(0, 24, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            pixels = np.clip(noise.astype(np.int16) + base, 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(root / label / f"{label}-{position}.png")
    return root


def test_manifest_build_reports_deterministic_splits(
    tmp_path: Path, images: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.jsonl"

    result = _run(capsys, "manifest-build", "--images", str(images), "--output", str(manifest))

    assert result["items"] == 12
    assert result["index"] + result["query"] == 12
    assert result["query"] > 0
    assert manifest.is_file()


def test_pca_evaluate_grid_and_query_share_one_store(
    tmp_path: Path, images: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    store_path = tmp_path / "pca.npz"
    projection_path = tmp_path / "projection.npz"
    _run(capsys, "manifest-build", "--images", str(images), "--output", str(manifest))

    embedded = _run(
        capsys,
        "embed-pca",
        "--manifest", str(manifest),
        "--image-root", str(images),
        "--output", str(store_path),
        "--components", "3",
        "--image-size", str(IMAGE_SIZE),
        "--projection-output", str(projection_path),
    )
    assert embedded["shape"] == [12, 3]
    assert Path(embedded["projection"]).is_file()

    evaluation_path = tmp_path / "evaluation.json"
    evaluation = _run(
        capsys,
        "evaluate",
        "--embeddings", str(store_path),
        "--k", "2",
        "--output", str(evaluation_path),
    )
    assert evaluation["k"] == 2
    assert evaluation["evaluated_queries"] > 0
    assert set(evaluation["per_class"]) == {"forest", "water"}
    assert json.loads(evaluation_path.read_text(encoding="utf-8")) == evaluation

    grid_path = tmp_path / "grid.png"
    grid = _run(
        capsys,
        "result-grid",
        "--embeddings", str(store_path),
        "--manifest", str(manifest),
        "--image-root", str(images),
        "--output", str(grid_path),
        "--k", "2",
        "--mode", "best",
    )
    assert grid_path.is_file()
    assert len(grid["queries"]) == 2

    stored = EmbeddingStore.load(store_path)
    item_id = next(
        identifier
        for identifier, split in zip(stored.ids, stored.splits, strict=True)
        if split == "query"
    )
    by_id = _run(capsys, "query", "--embeddings", str(store_path), "--item-id", item_id, "--k", "3")
    assert by_id["query"] == {"item_id": item_id}
    assert by_id["backend"] == "pca"
    assert len(by_id["results"]) == 3
    assert item_id not in [result["item_id"] for result in by_id["results"]]

    by_image = _run(
        capsys,
        "query",
        "--embeddings", str(store_path),
        "--image", str(images / item_id),
        "--projection", str(projection_path),
        "--k", "3",
    )
    # A query-split item is not part of the index, so re-embedding its file must
    # reproduce the stored ranking exactly. This is the guarantee that makes the
    # saved projection usable for images the corpus has never seen.
    assert by_image["query"] == {"image": str(images / item_id)}
    assert by_image["results"] == by_id["results"]


def test_query_requires_exactly_one_subject(tmp_path: Path) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["query", "--embeddings", str(tmp_path / "s.npz")])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["query", "--embeddings", str(tmp_path / "s.npz"), "--item-id", "a", "--image", "b"]
        )


def test_errors_are_reported_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["eovr", "manifest-build", "--images", str(tmp_path / "absent"), "--output", "out.jsonl"],
    )

    with pytest.raises(SystemExit) as failure:
        main()

    assert "error: image root does not exist" in str(failure.value)
