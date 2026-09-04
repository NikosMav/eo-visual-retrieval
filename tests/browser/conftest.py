"""Serve a tiny synthetic corpus through the installed CLI for browser tests."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pytest
from PIL import Image

from eo_visual_retrieval.embeddings.projection import PcaProjection
from eo_visual_retrieval.embeddings.store import EmbeddingStore
from eo_visual_retrieval.hashing import file_sha256
from eo_visual_retrieval.manifests import write_jsonl
from eo_visual_retrieval.models import ImageRecord, Split


def _prepare_corpus(root: Path) -> list[str]:
    """Return CLI arguments; all images, vectors, and basis are test fixtures."""

    rows: list[tuple[str, Split, str, tuple[int, int, int]]] = [
        ("forest/index-a.png", "index", "Forest", (30, 90, 20)),
        ("forest/index-b.png", "index", "Forest", (40, 80, 10)),
        ("water/index-c.png", "index", "SeaLake", (20, 30, 100)),
        ("forest/query-a.png", "query", "Forest", (50, 100, 30)),
        ("forest/query-b.png", "query", "Forest", (60, 110, 40)),
        ("water/query-c.png", "query", "SeaLake", (30, 40, 120)),
    ]
    records: list[ImageRecord] = []
    for item_id, split, label, color in rows:
        image = root / item_id
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=color).save(image)
        records.append(ImageRecord(
            item_id=item_id, path=item_id, split=split, label=label,
            metadata={"sha256": file_sha256(image)},
        ))
    manifest = root / "manifest.jsonl"
    write_jsonl(records, manifest)
    digest = file_sha256(manifest)
    projection = PcaProjection(
        mean=np.zeros(8 * 8 * 3, dtype=np.float32),
        components=np.eye(2, 8 * 8 * 3, dtype=np.float32),
        image_size=8,
        seed=42,
        metadata={"manifest_sha256": digest},
    )
    projection_path = root / "projection.npz"
    projection.save(projection_path)
    vectors = projection.embed_images([root / record.path for record in records])
    arguments = ["--manifest", str(manifest), "--image-root", str(root)]
    for backend, model in (("pca", "pca"), ("dinov2", "dinov2_vits14")):
        store_path = root / f"{backend}.npz"
        # The second store exercises multi-model UI behavior without a neural
        # encoder. These fabricated vectors are never benchmark evidence.
        EmbeddingStore(
            ids=tuple(record.item_id for record in records),
            vectors=vectors if backend == "pca" else vectors[:, ::-1],
            labels=tuple(record.label for record in records),
            splits=tuple(record.split for record in records),
            metadata={
                "backend": backend, "model": model, "manifest_sha256": digest,
                "image_size": 8, "synthetic_test_fixture": True,
            },
        ).save(store_path)
        arguments.extend(["--store", str(store_path)])
    return [*arguments, "--projection", str(projection_path), "--k", "2"]


def _wait_for_server(process: subprocess.Popen[bytes], url: str, log: Path) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"explorer exited before readiness:\n{log.read_text(encoding='utf-8')}"
            )
        try:
            with urlopen(f"{url}/healthz", timeout=1) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            time.sleep(0.05)
    raise RuntimeError(f"explorer did not become ready:\n{log.read_text(encoding='utf-8')}")


@pytest.fixture(scope="session")
def explorer_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    root = tmp_path_factory.mktemp("browser-corpus")
    arguments = _prepare_corpus(root)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    if environment.get("EOVR_TEST_WHEEL") == "1":
        environment.pop("PYTHONPATH", None)
    log = root / "server.log"
    with log.open("wb") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "eo_visual_retrieval.cli", "serve", *arguments,
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            url = f"http://127.0.0.1:{port}"
            _wait_for_server(process, url, log)
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
