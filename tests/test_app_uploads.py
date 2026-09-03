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
