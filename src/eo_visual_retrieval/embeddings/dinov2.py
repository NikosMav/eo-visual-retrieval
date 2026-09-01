"""Frozen DINOv2 image embeddings through the official PyTorch Hub entrypoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

ALLOWED_MODELS = {
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vits14_reg",
    "dinov2_vitb14_reg",
}


def _resolve_device(torch: object, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[attr-defined]


def dinov2_embeddings(
    paths: list[Path],
    *,
    model_name: str = "dinov2_vits14",
    batch_size: int = 8,
    device: str = "auto",
) -> NDArray[np.float32]:
    """Embed local RGB images and return L2-normalized CLS features."""

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"unsupported DINOv2 model: {model_name}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    try:
        import torch
        from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
    except ImportError as error:
        message = 'DINOv2 support is optional; install with pip install -e ".[ml]"'
        raise RuntimeError(message) from error

    selected_device = _resolve_device(torch, device)
    model = torch.hub.load(
        "facebookresearch/dinov2",
        model_name,
        trust_repo=True,
    )
    model.eval().to(selected_device)
    transform = Compose(
        [
            Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    batches: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            tensor = torch.stack(images).to(selected_device)
            features = model(tensor)
            features = torch.nn.functional.normalize(features, dim=1)
            batches.append(features.cpu().numpy().astype(np.float32, copy=False))

    if not batches:
        raise ValueError("at least one image is required")
    return np.concatenate(batches, axis=0)
