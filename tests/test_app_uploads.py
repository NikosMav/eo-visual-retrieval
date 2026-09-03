"""Uploaded bytes are hostile until proven otherwise."""

from __future__ import annotations

import io
import zlib

import numpy as np
import pytest
from PIL import Image

from eo_visual_retrieval.app.uploads import (
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_PIXELS,
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


def _png_with_declared_size(width: int, height: int) -> bytes:
    """Build a tiny PNG whose IHDR *declares* width/height without containing

    that many pixels: the classic decompression-bomb shape. The IHDR chunk's
    CRC is recomputed over the patched chunk so Pillow accepts the header
    (a stale CRC would make Pillow reject the file as unreadable instead of
    reaching the pixel-count check this is meant to exercise).
    """
    header = io.BytesIO()
    Image.new("RGB", (2, 2)).save(header, format="PNG")
    payload = bytearray(header.getvalue())
    # IHDR data (width, height, ...) starts right after the 8-byte PNG
    # signature, the 4-byte chunk length, and the 4-byte "IHDR" chunk type.
    payload[16:24] = width.to_bytes(4, "big") + height.to_bytes(4, "big")
    ihdr_chunk = bytes(payload[12:29])  # b"IHDR" + the 13 IHDR data bytes
    payload[29:33] = zlib.crc32(ihdr_chunk).to_bytes(4, "big")
    return bytes(payload)


def test_declared_pixel_bomb_is_refused() -> None:
    # A tiny file that declares a canvas comfortably over the lowered
    # MAX_UPLOAD_PIXELS (4096x4096), but still under Pillow's own default
    # decompression-bomb ceiling, so this trips *our* check rather than
    # Pillow's "broken PNG" path.
    width = height = 4_500
    assert width * height > MAX_UPLOAD_PIXELS

    payload = _png_with_declared_size(width, height)

    with pytest.raises(ValueError, match=f"{MAX_UPLOAD_PIXELS}-pixel limit"):
        decode_upload(payload, image_size=IMAGE_SIZE)


def test_image_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="image_size"):
        decode_upload(_png(), image_size=0)
