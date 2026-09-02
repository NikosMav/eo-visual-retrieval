"""Frozen SSL4EO-S12 embeddings for the EuroSAT multispectral benchmark."""

from __future__ import annotations

import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eo_visual_retrieval.datasets.eurosat import (
    EUROSAT_ARCHIVE_MD5,
    EUROSAT_BAND_ORDER,
    archive_members,
    band_indices,
    read_archive_member,
    verify_archive,
)
from eo_visual_retrieval.hashing import file_sha256, verify_sha256
from eo_visual_retrieval.models import ImageRecord

__all__ = [
    "CHECKPOINT_SHA256",
    "MODEL_NAME",
    "SSL4EO_BAND_INDICES",
    "SSL4EO_BAND_ORDER",
    "file_sha256",
    "prepare_multispectral",
    "ssl4eo_embeddings",
    "verify_sha256",
]

MODEL_NAME = "ssl4eo-s12-moco-resnet50"
CHECKPOINT_REVISION = "da4f3c9dbe09272eb902f3b37f46635fa4726879"
CHECKPOINT_FILENAME = "resnet50_sentinel2_all_moco-df8b932e.pth"
CHECKPOINT_SHA256 = "df8b932e2a23a0773febedf3f650aa7d342b805f7876ca5ed6b139d7245d7c09"
CHECKPOINT_REPOSITORY = "torchgeo/resnet50_sentinel2_all_moco"

# The archive puts B8A last; SSL4EO-S12 expects it between B8 and B9.
SSL4EO_BAND_ORDER = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)
SSL4EO_BAND_INDICES = band_indices(SSL4EO_BAND_ORDER)


def prepare_multispectral(source: NDArray[np.generic]) -> NDArray[np.float32]:
    """Reorder EuroSAT bands and apply the registered SSL4EO-S12 scaling.

    EuroSAT stores Sentinel-2 Level-1C digital numbers. The pretrained model's
    registered preprocessing clips those values to the reflectance-like range
    0..10000 and divides by 10000. Spatial resizing happens later in torch so a
    batch can be transferred to the selected device only once.
    """

    if source.ndim != 3 or source.shape[0] != len(EUROSAT_BAND_ORDER):
        raise ValueError(
            "EuroSAT multispectral input must have shape (13, height, width)"
        )
    ordered = source[np.asarray(SSL4EO_BAND_INDICES)]
    return (np.clip(ordered, 0, 10_000).astype(np.float32) / 10_000.0).astype(
        np.float32,
        copy=False,
    )


def _checkpoint_state(payload: Any) -> dict[str, Any]:
    """Extract either a TorchGeo-converted or original MoCo encoder state."""

    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is not a state dictionary")
    candidate: Mapping[str, Any] = payload
    nested = payload.get("state_dict")
    if isinstance(nested, Mapping):
        candidate = nested

    state: dict[str, Any] = {}
    moco_prefix = "module.encoder_q."
    is_original_moco = any(str(key).startswith(moco_prefix) for key in candidate)
    for raw_key, value in candidate.items():
        key = str(raw_key)
        if is_original_moco and not key.startswith(moco_prefix):
            continue
        if is_original_moco:
            key = key.removeprefix(moco_prefix)
        elif key.startswith("module."):
            key = key.removeprefix("module.")
        if is_original_moco and key.startswith("fc."):
            continue
        state[key] = value
    return state


def _load_model(torch: Any, checkpoint: Path, device: str) -> Any:
    try:
        from torchvision.models import resnet50
    except ImportError as error:  # pragma: no cover - environment-dependent
        message = 'SSL4EO-S12 support is optional; install with pip install -e ".[ml,geo]"'
        raise RuntimeError(message) from error

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = _checkpoint_state(payload)
    convolution = state.get("conv1.weight")
    if convolution is None or tuple(convolution.shape) != (64, 13, 7, 7):
        raise ValueError("checkpoint is not the expected 13-band ResNet-50 encoder")

    model = resnet50(weights=None)
    model.conv1 = torch.nn.Conv2d(13, 64, kernel_size=7, stride=2, padding=3, bias=False)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_classifier_keys = {"fc.weight", "fc.bias"}
    missing = set(incompatible.missing_keys) - allowed_classifier_keys
    unexpected = set(incompatible.unexpected_keys) - allowed_classifier_keys
    if missing or unexpected:
        raise ValueError(
            "checkpoint does not match ResNet-50; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    model.fc = torch.nn.Identity()
    return model.eval().to(device)


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def ssl4eo_embeddings(
    records: Sequence[ImageRecord],
    *,
    archive: Path,
    checkpoint: Path,
    batch_size: int = 16,
    device: str = "auto",
    expected_archive_md5: str | None = EUROSAT_ARCHIVE_MD5,
    expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
) -> NDArray[np.float32]:
    """Embed selected EuroSAT members with a frozen 13-band SSL4EO encoder."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    verify_archive(archive, expected_md5=expected_archive_md5)
    verify_sha256(checkpoint, expected_checkpoint_sha256)
    members = archive_members(records)

    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment-dependent
        message = 'SSL4EO-S12 support is optional; install with pip install -e ".[ml,geo]"'
        raise RuntimeError(message) from error

    selected_device = _resolve_device(torch, device)
    model = _load_model(torch, checkpoint, selected_device)
    batches: list[NDArray[np.float32]] = []
    with zipfile.ZipFile(archive) as bundle, torch.inference_mode():
        for start in range(0, len(members), batch_size):
            selected = members[start : start + batch_size]
            arrays = [
                prepare_multispectral(read_archive_member(bundle, member))
                for member in selected
            ]
            tensor = torch.from_numpy(np.stack(arrays)).to(selected_device)
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            tensor = tensor[:, :, 16:240, 16:240]
            features = torch.nn.functional.normalize(model(tensor), dim=1)
            batches.append(features.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(batches, axis=0)
