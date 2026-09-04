"""Frozen DINOv2 image embeddings through the official PyTorch Hub entrypoint.

Torch Hub resolves a bare repository name to its default branch and then reuses
whatever it downloaded first, so a run records no usable code identity: locking
Python packages does not pin downloaded executable model code. This module
requests one immutable commit and verifies the bytes that arrive, because a
reference states what was asked for and only a digest states what was received.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from eo_visual_retrieval.hashing import source_tree_sha256

ALLOWED_MODELS = {
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vits14_reg",
    "dinov2_vitb14_reg",
}

# The commit recovered from the Torch Hub cache that produced the published
# EuroSAT vectors, and the digest of that extracted tree. facebookresearch/dinov2
# publishes no tags, so a commit is the only immutable reference available.
HUB_REPO = "facebookresearch/dinov2"
HUB_REF = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
HUB_TREE_SHA256 = "735a0c2c248537a5a746d87d86ea4b9a32c869ec0ffcc470b05e90be3c1246a8"

# Bytecode caches appear only after the code is imported, so the same download
# would otherwise digest differently on its second use.
_GENERATED_DIRECTORIES = frozenset({"__pycache__"})


def hub_provenance() -> dict[str, str]:
    """Return the code identity a run should record beside its vectors."""

    return {
        "hub_repo": HUB_REPO,
        "hub_ref": HUB_REF,
        "hub_tree_sha256": HUB_TREE_SHA256,
    }


def _resolve_device(torch: object, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[attr-defined]


def _hub_source_root(torch: object) -> Path:
    """Return the directory Torch Hub extracts the pinned commit into."""

    owner, repository = HUB_REPO.split("/")
    hub_dir = torch.hub.get_dir()  # type: ignore[attr-defined]
    return Path(hub_dir) / f"{owner}_{repository}_{HUB_REF}"


def _verify_hub_source(torch: object) -> None:
    """Refuse to embed with model code that is not the pinned commit."""

    root = _hub_source_root(torch)
    if not root.is_dir():
        raise RuntimeError(
            f"cannot verify pinned DINOv2 code: {root} is missing. Torch Hub chooses "
            "this directory name, so a layout change needs a new check rather than an "
            "unverified run"
        )
    observed = source_tree_sha256(root, skip_directories=_GENERATED_DIRECTORIES)
    if observed != HUB_TREE_SHA256:
        raise RuntimeError(
            f"DINOv2 code at {HUB_REF} hashed to {observed}, expected {HUB_TREE_SHA256}; "
            "refusing to embed with model code that is not the reviewed commit"
        )


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
    # Checked before the checkpoint is fetched: downloading a model to discover
    # there is nothing to embed wastes minutes on a first run.
    if not paths:
        raise ValueError("at least one image is required")

    try:
        import torch
        from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
    except ImportError as error:
        message = 'DINOv2 support is optional; install with pip install -e ".[ml]"'
        raise RuntimeError(message) from error

    selected_device = _resolve_device(torch, device)
    model = torch.hub.load(
        f"{HUB_REPO}:{HUB_REF}",
        model_name,
        trust_repo=True,
    )
    _verify_hub_source(torch)
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

    return np.concatenate(batches, axis=0)
