"""Frozen SSL4EO-S12 embeddings for the EuroSAT multispectral benchmark."""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eo_visual_retrieval.benchmarks.eurosat import (
    EUROSAT_ARCHIVE_MD5,
    EUROSAT_SOURCE,
    file_md5,
)
from eo_visual_retrieval.models import ImageRecord

MODEL_NAME = "ssl4eo-s12-moco-resnet50"
CHECKPOINT_REVISION = "da4f3c9dbe09272eb902f3b37f46635fa4726879"
CHECKPOINT_FILENAME = "resnet50_sentinel2_all_moco-df8b932e.pth"
CHECKPOINT_SHA256 = "df8b932e2a23a0773febedf3f650aa7d342b805f7876ca5ed6b139d7245d7c09"
CHECKPOINT_REPOSITORY = "torchgeo/resnet50_sentinel2_all_moco"

# The archive puts B8A last; SSL4EO-S12 expects it between B8 and B9.
EUROSAT_BAND_ORDER = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B10",
    "B11",
    "B12",
    "B8A",
)
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
SSL4EO_BAND_INDICES = tuple(EUROSAT_BAND_ORDER.index(band) for band in SSL4EO_BAND_ORDER)


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for a local model checkpoint."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> str:
    """Validate a checkpoint identity before deserializing it."""

    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")
    observed = file_sha256(path)
    if observed.lower() != expected.lower():
        raise ValueError(
            f"checkpoint checksum mismatch: expected {expected}, observed {observed}"
        )
    return observed


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


def _archive_members(records: Sequence[ImageRecord]) -> list[str]:
    if not records:
        raise ValueError("at least one image record is required")
    members: list[str] = []
    for record in records:
        if record.source != EUROSAT_SOURCE:
            raise ValueError(f"record is not a {EUROSAT_SOURCE} item: {record.item_id}")
        member = record.metadata.get("archive_member")
        if not isinstance(member, str) or not member:
            raise ValueError(f"record lacks an archive member: {record.item_id}")
        members.append(member)
    if len(set(members)) != len(members):
        raise ValueError("manifest contains duplicate EuroSAT archive members")
    return members


def _read_member(bundle: zipfile.ZipFile, member: str) -> NDArray[np.float32]:
    try:
        from rasterio.io import MemoryFile
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("SSL4EO-S12 support requires the 'geo' dependency group") from error

    try:
        payload = bundle.read(member)
    except KeyError as error:
        raise ValueError(f"EuroSAT archive member does not exist: {member}") from error
    with MemoryFile(payload) as memory_file, memory_file.open() as dataset:
        if dataset.count != len(EUROSAT_BAND_ORDER):
            raise ValueError(f"EuroSAT archive member does not have 13 bands: {member}")
        source = dataset.read()
    return prepare_multispectral(source)


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
    if not archive.is_file():
        raise ValueError(f"EuroSAT archive does not exist: {archive}")
    if expected_archive_md5 is not None:
        observed_archive_md5 = file_md5(archive)
        if observed_archive_md5.lower() != expected_archive_md5.lower():
            raise ValueError(
                "EuroSAT archive checksum mismatch: "
                f"expected {expected_archive_md5}, observed {observed_archive_md5}"
            )
    verify_sha256(checkpoint, expected_checkpoint_sha256)
    members = _archive_members(records)

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
            arrays = [_read_member(bundle, member) for member in selected]
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
