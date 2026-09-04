"""Browser regressions against a served synthetic corpus, optionally installed wheel.

Install the ``app,browser`` extras and Chromium before explicitly running this
directory. Ordinary lightweight runs skip this module when Playwright is absent.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from PIL import Image

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import Page, expect  # noqa: E402


def test_installed_wheel_has_no_model_dependencies() -> None:
    if os.environ.get("EOVR_TEST_WHEEL") != "1":
        pytest.skip("wheel isolation is checked in the dedicated CI job")
    import eo_visual_retrieval

    assert eo_visual_retrieval.__file__ is not None
    assert Path(eo_visual_retrieval.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    for module in ("torch", "torchvision", "terratorch", "sklearn"):
        assert importlib.util.find_spec(module) is None, f"unexpected serving dependency: {module}"


def test_class_filter_and_comparison_navigation(page: Page, explorer_url: str) -> None:
    page.goto(explorer_url)
    classes = page.get_by_label("Scene class", exact=True)
    picker = page.get_by_label("Corpus query", exact=True)
    expect(classes).to_be_visible()
    expect(page.get_by_role("heading", name="PCA", exact=True)).to_be_visible()
    expect(page.get_by_role("heading", name="DINOv2 ViT-S/14", exact=True)).to_be_visible()

    classes.select_option("SeaLake")
    expect(picker.locator("option")).to_have_count(1)
    expect(picker).to_have_value("water/query-c.png")
    page.get_by_role("button", name="Compare representations").click()
    expect(page.get_by_alt_text("Selected satellite query: SeaLake")).to_be_visible()
    assert parse_qs(urlsplit(page.url).query) == {"item_id": ["water/query-c.png"]}
    expect(classes).to_have_value("SeaLake")

    classes.select_option("")
    expect(picker.locator("option")).to_have_count(3)
    picker.select_option("forest/query-b.png")
    page.get_by_role("button", name="Compare representations").click()
    expect(page.get_by_alt_text("Selected satellite query: Forest")).to_be_visible()
    assert parse_qs(urlsplit(page.url).query) == {"item_id": ["forest/query-b.png"]}
    # These are real HTTP images from the installed package's thumbnail route.
    expect(page.locator(".query-image")).to_have_js_property("naturalWidth", 8)


def test_upload_has_unknown_relevance_and_only_pca(page: Page, explorer_url: str) -> None:
    page.goto(explorer_url)
    page.get_by_text("Try your own image", exact=True).click()
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=(80, 120, 40)).save(buffer, format="PNG")
    page.get_by_label("Image file", exact=True).set_input_files({
        "name": "synthetic-chip.png", "mimeType": "image/png", "buffer": buffer.getvalue(),
    })
    page.get_by_role("button", name="Rank upload").click()
    expect(page).to_have_url(f"{explorer_url}/compare/upload")
    expect(page.get_by_role("heading", name="PCA", exact=True)).to_be_visible()
    expect(page.locator("article.row")).to_have_count(1)
    expect(page.get_by_role("heading", name="DINOv2 ViT-S/14", exact=True)).to_have_count(0)
    expect(page.locator(".results .notice")).to_contain_text("no label")
    expect(page.locator(".tile .relevance")).to_have_text([
        "Unknown relevance", "Unknown relevance",
    ])
    expect(page.locator(".tile.hit, .tile.miss")).to_have_count(0)


def test_malformed_image_shows_html_error_and_recovery(page: Page, explorer_url: str) -> None:
    page.goto(explorer_url)
    page.get_by_text("Try your own image", exact=True).click()
    page.get_by_label("Image file", exact=True).set_input_files({
        "name": "broken.png", "mimeType": "image/png", "buffer": b"not an image",
    })
    with page.expect_response("**/compare/upload") as response:
        page.get_by_role("button", name="Rank upload").click()
    assert response.value.status == 400
    assert "text/html" in response.value.headers["content-type"]
    expect(page.get_by_role("alert")).to_contain_text("not a readable image")
    page.get_by_role("link", name="Return to the explorer").click()
    expect(page.get_by_role("heading", name="PCA", exact=True)).to_be_visible()


def test_narrow_layout_and_keyboard_controls(page: Page, explorer_url: str) -> None:
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(explorer_url)
    expect(page.locator(".workspace")).to_have_css("display", "grid")
    page.keyboard.press("Tab")
    expect(page.get_by_role("link", name="Skip to results")).to_be_focused()
    expect(page.get_by_role("link", name="Skip to results")).to_be_visible()

    classes = page.get_by_label("Scene class", exact=True)
    classes.focus()
    page.keyboard.press("Tab")
    expect(page.get_by_label("Corpus query", exact=True)).to_be_focused()
    page.keyboard.press("Tab")
    expect(page.get_by_role("button", name="Compare representations")).to_be_focused()
    page.keyboard.press("Enter")
    expect(page.get_by_alt_text("Selected satellite query: Forest")).to_be_visible()

    provenance = page.locator("details.provenance").first
    provenance.locator("summary").focus()
    page.keyboard.press("Enter")
    expect(provenance).to_have_attribute("open", "")
    expect(provenance).to_contain_text("manifest sha256")
    expect(provenance).to_contain_text("exact-cosine")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
