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
