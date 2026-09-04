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
