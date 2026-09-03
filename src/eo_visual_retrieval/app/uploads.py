"""Decode an uploaded image, or refuse it.

This surface is intended to be publicly reachable, so incoming bytes are treated
as hostile: they are size-capped before any decode is attempted, never written to
disk, and their type is established by decoding rather than by trusting a
declared content type.
"""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_UPLOAD_PIXELS = 64_000_000
MAX_IMAGE_SIZE = 1024


def decode_upload(data: bytes, *, image_size: int) -> NDArray[np.float32]:
    """Return one flattened, 0-1 scaled RGB row from uploaded bytes."""

    if image_size < 1:
        raise ValueError("image_size must be positive")
    if image_size > MAX_IMAGE_SIZE:
        raise ValueError(f"image_size exceeds the {MAX_IMAGE_SIZE} limit")
    if not data:
        raise ValueError("uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"uploaded file exceeds the {MAX_UPLOAD_BYTES}-byte limit")

    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            width, height = source.size
            if width * height > MAX_UPLOAD_PIXELS:
                raise ValueError(f"uploaded image exceeds the {MAX_UPLOAD_PIXELS}-pixel limit")
            resized = source.convert("RGB").resize((image_size, image_size))
            pixels = np.asarray(resized, dtype=np.float32) / 255.0
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("uploaded file is not a readable image") from error

    return pixels.reshape(1, -1)
