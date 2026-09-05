"""Browser behavior for multimodal forms; vectors are fixtures, never quality evidence."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

pytest.importorskip("playwright")
import uvicorn  # noqa: E402
from playwright.sync_api import Page, expect  # noqa: E402

from eo_visual_retrieval.app.search import create_search_app  # noqa: E402
from eo_visual_retrieval.embeddings.store import EmbeddingStore  # noqa: E402
from eo_visual_retrieval.models import ImageRecord  # noqa: E402
from eo_visual_retrieval.multimodal import MultimodalSearch  # noqa: E402


class BrowserEncoder:
    metadata: dict[str, Any] = {"model": "synthetic browser fixture", "dimension": 2}

    def encode_text(self, text: str) -> NDArray[np.float32]:
        return np.array([1, 0], dtype=np.float32)

    def encode_image(self, image: Image.Image) -> NDArray[np.float32]:
        return np.array([0, 1], dtype=np.float32)


@pytest.fixture
def search_url(tmp_path: Path) -> Iterator[str]:
    records = []
    for item_id, color in (("a", "green"), ("b", "blue"), ("c", "gray")):
        Image.new("RGB", (32, 32), color).save(tmp_path / f"{item_id}.png")
        records.append(ImageRecord(item_id, f"{item_id}.png", "index"))
    store = EmbeddingStore(
        ("a", "b", "c"),
        np.array([[1, 0], [0, 1], [1, 1]], np.float32),
        (None,) * 3,
        ("index",) * 3,
        BrowserEncoder.metadata,
    )
    engine = MultimodalSearch(records, store, BrowserEncoder(), image_root=tmp_path)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_search_app(engine), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_search_modes_and_example_selection(page: Page, search_url: str, tmp_path: Path) -> None:
    page.goto(search_url)
    page.get_by_label("Describe the scene").fill("urban buildings")
    page.get_by_role("button", name="Search scenes").click()
    expect(page.get_by_role("status", name="Search progress")).to_contain_text("text search")
    expect(page.locator(".result")).to_have_count(3)
    page.get_by_role("button", name="Use as example").first.click()
    page.get_by_role("button", name="Search scenes").click()
    expect(page.get_by_role("status", name="Search progress")).to_contain_text("hybrid search")
    expect(page.locator(".result")).to_have_count(2)
    upload = tmp_path / "upload.png"
    Image.new("RGB", (32, 16)).save(upload)
    page.get_by_label("Upload an RGB image").set_input_files(upload)
    expect(page.get_by_label("Or select a corpus example")).to_have_value("")
    page.get_by_label("Describe the scene").fill("")
    page.get_by_role("button", name="Search scenes").click()
    expect(page.get_by_role("status", name="Search progress")).to_contain_text("image search")
    expect(page.locator(".result")).to_have_count(3)


def test_plan_empty_state_and_mobile_layout(page: Page, search_url: str) -> None:
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(search_url)
    page.get_by_label("Describe the scene").fill(
        "Sentinel imagery showing recent urban expansion near Athens with low cloud coverage."
    )
    page.get_by_role("button", name="Review filters").click()
    expect(page.locator("#filters")).to_contain_text("sentinel-2-l2a")
    expect(page.locator("#notes")).to_contain_text("Single-scene")
    page.get_by_role("button", name="Search scenes").click()
    expect(page.get_by_role("status", name="Search progress")).to_contain_text("No index scenes")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.get_by_text("Place, date and cloud filters", exact=True).click()
    page.get_by_label("WGS84 bounding box", exact=True).fill("bad bounds")
    page.get_by_role("button", name="Review filters").click()
    expect(page.get_by_role("status", name="Search progress")).to_contain_text(
        "four bounding-box coordinates"
    )


def test_untrusted_result_ids_are_text(page: Page, search_url: str) -> None:
    """The browser never turns API strings into markup."""
    page.route(
        "**/api/search",
        lambda route: route.fulfill(
            content_type="application/json",
            body=json.dumps(
                {
                    "mode": "text",
                    "candidate_count": 1,
                    "plan": {"filters": {}, "notes": []},
                    "results": [
                        {
                            "item_id": '<img src=x onerror="window.injected=true">',
                            "score": 0.5,
                            "text_score": 0.5,
                            "image_score": None,
                            "metadata": {"date": None, "cloud_cover": None},
                        }
                    ],
                }
            ),
        ),
    )
    page.goto(search_url)
    page.get_by_label("Describe the scene").fill("urban")
    page.get_by_role("button", name="Search scenes").click()
    expect(page.locator(".result h3")).to_contain_text("<img src=x")
    assert page.evaluate("window.injected !== true")


def test_transparency_export_and_product_navigation(page: Page, search_url: str) -> None:
    page.goto(search_url)
    page.get_by_label("Describe the scene").fill("urban buildings")
    page.get_by_role("button", name="Search scenes").click()
    expect(page.locator("#diagnostics")).to_contain_text("3 eligible")
    page.get_by_text("Why rank #1?", exact=True).click()
    expect(page.locator(".result").first).to_contain_text("Text alone: #1")
    with page.expect_download() as downloaded:
        page.get_by_role("button", name="Export search record").click()
    record = json.loads(Path(downloaded.value.path()).read_text(encoding="utf-8"))
    assert record["diagnostics"]["excluded_by_filters"] == 0
    assert record["results"][0]["explanation"]["text_contribution"] == 1
    page.set_viewport_size({"width": 390, "height": 844})
    for name, heading in (("Findings", "What the evidence says."),
                          ("Data & experiments", "Better evidence.")):
        page.get_by_role("navigation", name="Product").get_by_role("link", name=name).click()
        expect(page.get_by_role("heading", level=1)).to_contain_text(heading)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
