"""Pinned, frozen RemoteCLIP RGB/text inference in one shared coordinate system."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from eo_visual_retrieval.hashing import file_sha256
from eo_visual_retrieval.vectors import l2_normalize

REPOSITORY = "chendelong/RemoteCLIP"
REVISION = "bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"
CHECKPOINT = "RemoteCLIP-ViT-B-32.pt"
CHECKPOINT_SHA256 = "60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4"
OPEN_CLIP_VERSION = "3.3.0"
DIMENSION = 512


def space_metadata() -> dict[str, Any]:
    """The complete supported embedding-space identity, excluding corpus/device."""
    return {
        "backend": "remoteclip",
        "model": "RemoteCLIP-ViT-B-32",
        "model_repository": REPOSITORY,
        "model_revision": REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "open_clip_version": OPEN_CLIP_VERSION,
        "preprocessing": "openclip-ViT-B-32-default-rgb-v1",
        "dimension": DIMENSION,
        "normalized": True,
    }


class MultimodalEncoder(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def encode_text(self, text: str) -> NDArray[np.float32]: ...

    def encode_image(self, image: Image.Image) -> NDArray[np.float32]: ...


class RemoteClipEncoder:
    """Load once per process; never execute code from the model repository.

    Download is allowed only by the offline embedding command. Serving requires
    the pinned checkpoint in the HF cache (or a local, hash-verified copy).
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        checkpoint: Path | None = None,
        allow_download: bool = False,
    ) -> None:
        try:
            import open_clip
            import torch
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                'install the multimodal extra: pip install -e ".[multimodal]"'
            ) from error
        if version("open-clip-torch") != OPEN_CLIP_VERSION:
            raise ValueError(f"RemoteCLIP requires open-clip-torch=={OPEN_CLIP_VERSION}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu, cuda, or auto")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if checkpoint is None:
            try:
                checkpoint = Path(
                    hf_hub_download(
                        REPOSITORY,
                        CHECKPOINT,
                        revision=REVISION,
                        local_files_only=not allow_download,
                    )
                )
            except Exception as error:
                raise RuntimeError(
                    "Pinned RemoteCLIP checkpoint unavailable; run embed-remoteclip first "
                    "or supply --checkpoint. Serving never downloads weights."
                ) from error
        if not checkpoint.is_file() or file_sha256(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("RemoteCLIP checkpoint SHA-256 does not match the pinned weights")
        self._torch = torch
        self._device = device
        self._model, _, self._preprocess = open_clip.create_model_and_transforms("ViT-B-32")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state, strict=True)
        self._model = self._model.to(device).eval()
        self._model.requires_grad_(False)
        self._tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @property
    def metadata(self) -> dict[str, Any]:
        return space_metadata()

    def encode_text(self, text: str) -> NDArray[np.float32]:
        text = text.strip()
        if not text or len(text) > 1000:
            raise ValueError("text must contain 1 to 1000 characters")
        # OpenCLIP otherwise truncates silently. Two positions are BOS/EOS.
        if len(self._tokenizer.encode(text)) > 75:
            raise ValueError("description exceeds 75 CLIP tokens; shorten the visual description")
        with self._torch.inference_mode():
            output = self._model.encode_text(self._tokenizer([text]).to(self._device))
        return l2_normalize(output.float().cpu().numpy())[0]

    def encode_image(self, image: Image.Image) -> NDArray[np.float32]:
        with self._torch.inference_mode():
            pixels = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
            output = self._model.encode_image(pixels)
        return l2_normalize(output.float().cpu().numpy())[0]

    def encode_paths(self, paths: Sequence[Path], *, batch_size: int = 16) -> NDArray[np.float32]:
        if not paths or not 1 <= batch_size <= 256:
            raise ValueError("provide images and a batch size between 1 and 256")
        batches = []
        for start in range(0, len(paths), batch_size):
            inputs = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as source:
                    inputs.append(self._preprocess(source.convert("RGB")))
            with self._torch.inference_mode():
                pixels = self._torch.stack(inputs).to(self._device)
                output = self._model.encode_image(pixels)
            batches.append(output.float().cpu().numpy())
        return l2_normalize(np.concatenate(batches))
