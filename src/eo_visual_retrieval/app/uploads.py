"""Decode an uploaded image, or refuse it.

This surface is intended to be publicly reachable, so incoming bytes are treated
as hostile: they are size-capped before any decode is attempted, and their type
is established by decoding rather than by trusting a declared content type.

The decode itself works entirely in memory. It is not true, however, that the
bytes never touch disk on the served path: the pinned Starlette multipart parser
spools any file part larger than 1 MB to an on-disk temporary file before this
module ever sees the data. The request-size guard in ``app.main`` buffers and
bounds actual bytes before form parsing, including bodies with no declared
length, so an oversized request never reaches the temporary-file spool.
"""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
# Every image is immediately resized to a basis of 64x64 pixels or smaller, so
# 4096x4096 (16,777,216 pixels) is already generous headroom: it bounds both the
# allocation a hostile declared canvas can force and the CPU spent decoding it,
# without constraining any real upload this surface would ever usefully accept.
MAX_UPLOAD_PIXELS = 16_777_216
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
