"""Frozen SSL4EO-S12 embeddings for the EuroSAT multispectral benchmark.

Two variants of the same pretrained encoder are supported. They share an
architecture, a pretraining corpus, and a preprocessing rule, and differ only
in how many Sentinel-2 bands they consume. Running both over identical patches
isolates the input bands as the single variable, which is the controlled
ablation ADR 0003 recorded as missing and ADR 0009 pre-registered.
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    "SSL4EO_ALL",
    "SSL4EO_BAND_INDICES",
    "SSL4EO_BAND_ORDER",
    "SSL4EO_RGB",
    "Ssl4eoVariant",
    "file_sha256",
    "prepare_bands",
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

# torchgeo registers this checkpoint's bands as ['B4', 'B3', 'B2'] — red first,
# not the blue-first order a casual reading of "RGB" might suggest.
SSL4EO_RGB_BAND_ORDER = ("B04", "B03", "B02")

RGB_MODEL_NAME = "ssl4eo-s12-rgb-moco-resnet50"
RGB_CHECKPOINT_REPOSITORY = "torchgeo/resnet50_sentinel2_rgb_moco"
RGB_CHECKPOINT_REVISION = "e6704867d1bf7f77c403d8078f41ccf5b2ffaa6c"
RGB_CHECKPOINT_FILENAME = "resnet50_sentinel2_rgb_moco-2b57ba8b.pth"
RGB_CHECKPOINT_SHA256 = (
    "2b57ba8b9964dbe1c409aac1bb79b4d97c19c874ffe7934799b7c8ad94ff85f0"
)


@dataclass(frozen=True)
class Ssl4eoVariant:
    """One pinned SSL4EO-S12 encoder and the archive bands it consumes."""

    model_name: str
    band_order: tuple[str, ...]
    checkpoint_repository: str
    checkpoint_revision: str
    checkpoint_filename: str
    checkpoint_sha256: str

    @property
    def band_indices(self) -> tuple[int, ...]:
        return band_indices(self.band_order)

    @property
    def channels(self) -> int:
        return len(self.band_order)


SSL4EO_ALL = Ssl4eoVariant(
    model_name=MODEL_NAME,
    band_order=SSL4EO_BAND_ORDER,
    checkpoint_repository=CHECKPOINT_REPOSITORY,
    checkpoint_revision=CHECKPOINT_REVISION,
    checkpoint_filename=CHECKPOINT_FILENAME,
    checkpoint_sha256=CHECKPOINT_SHA256,
)

SSL4EO_RGB = Ssl4eoVariant(
    model_name=RGB_MODEL_NAME,
    band_order=SSL4EO_RGB_BAND_ORDER,
    checkpoint_repository=RGB_CHECKPOINT_REPOSITORY,
    checkpoint_revision=RGB_CHECKPOINT_REVISION,
    checkpoint_filename=RGB_CHECKPOINT_FILENAME,
    checkpoint_sha256=RGB_CHECKPOINT_SHA256,
)


def prepare_bands(
    source: NDArray[np.generic], indices: Sequence[int]
) -> NDArray[np.float32]:
    """Select archive bands in a model's expected order and scale them.

    EuroSAT stores Sentinel-2 Level-1C digital numbers. torchgeo's registered
    transform for these weights divides by 10000 without clipping; this project
    additionally clips to 0..10000, because values above that are outside the
    nominal reflectance range. On EuroSAT the clip is very nearly inert — one
    value in 4.26 million exceeded 10000 across a sampled 80 patches — but it is
    a deliberate deviation from the registered transform, not a restatement of
    it, and it is recorded in docs/validation.md.

    Every variant shares this function, so a band comparison between them cannot
    be contaminated by a preprocessing difference. Spatial resizing happens later
    in torch so a batch transfers to the device only once.
    """

    if source.ndim != 3 or source.shape[0] != len(EUROSAT_BAND_ORDER):
        raise ValueError(
            "EuroSAT multispectral input must have shape (13, height, width)"
        )
    ordered = source[np.asarray(indices)]
    return (np.clip(ordered, 0, 10_000).astype(np.float32) / 10_000.0).astype(
        np.float32,
        copy=False,
    )


def prepare_multispectral(source: NDArray[np.generic]) -> NDArray[np.float32]:
    """Prepare all 13 bands for the multispectral variant."""

    return prepare_bands(source, SSL4EO_BAND_INDICES)


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


def _load_model(torch: Any, checkpoint: Path, device: str, channels: int) -> Any:
    try:
        from torchvision.models import resnet50
    except ImportError as error:  # pragma: no cover - environment-dependent
        message = 'SSL4EO-S12 support is optional; install with pip install -e ".[ml,geo]"'
        raise RuntimeError(message) from error

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = _checkpoint_state(payload)
    convolution = state.get("conv1.weight")
    if convolution is None or tuple(convolution.shape) != (64, channels, 7, 7):
        raise ValueError(
            f"checkpoint is not the expected {channels}-band ResNet-50 encoder"
        )

    model = resnet50(weights=None)
    model.conv1 = torch.nn.Conv2d(
        channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
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
    variant: Ssl4eoVariant = SSL4EO_ALL,
    expected_archive_md5: str | None = EUROSAT_ARCHIVE_MD5,
    expected_checkpoint_sha256: str | None = None,
) -> NDArray[np.float32]:
    """Embed selected EuroSAT members with a frozen SSL4EO encoder.

    The default variant is the 13-band multispectral encoder, so existing callers
    keep their exact behaviour.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    verify_archive(archive, expected_md5=expected_archive_md5)
    verify_sha256(checkpoint, expected_checkpoint_sha256 or variant.checkpoint_sha256)
    members = archive_members(records)

    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment-dependent
        message = 'SSL4EO-S12 support is optional; install with pip install -e ".[ml,geo]"'
        raise RuntimeError(message) from error

    selected_device = _resolve_device(torch, device)
    model = _load_model(torch, checkpoint, selected_device, variant.channels)
    band_selection = variant.band_indices
    batches: list[NDArray[np.float32]] = []
    with zipfile.ZipFile(archive) as bundle, torch.inference_mode():
        for start in range(0, len(members), batch_size):
            selected = members[start : start + batch_size]
            arrays = [
                prepare_bands(read_archive_member(bundle, member), band_selection)
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
