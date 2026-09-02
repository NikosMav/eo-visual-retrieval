"""Pinned, frozen TerraMind-Tiny S2L1C regression experiment for EuroSAT."""

from __future__ import annotations

import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eo_visual_retrieval.datasets.eurosat import (
    EUROSAT_ARCHIVE_MD5,
    archive_members,
    band_indices,
    read_archive_member,
    verify_archive,
)
from eo_visual_retrieval.hashing import verify_sha256
from eo_visual_retrieval.models import ImageRecord

MODEL_NAME = "terramind_v1_tiny"
CHECKPOINT_REPOSITORY = "ibm-esa-geospatial/TerraMind-1.0-tiny"
CHECKPOINT_REVISION = "2b5ac0a3ed7dd7e922ccfd595b56607f342df343"
CHECKPOINT_FILENAME = "TerraMind_v1_tiny.pt"
CHECKPOINT_SHA256 = "e56ea9ebcd4451078b9ca4893d5cd8a89bbee376ae16829c3e7fbbbc76de0eba"
BAND_ORDER = (
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
BAND_INDICES = band_indices(BAND_ORDER)
MODALITY_KEY = "untok_sen2l1c@224"


def prepare_multispectral(
    source: NDArray[np.generic], mean: Sequence[float], std: Sequence[float]
) -> NDArray[np.float32]:
    """Reorder raw L1C digital numbers, then use fixed pretraining statistics.

    Unlike SSL4EO preprocessing, do not divide by 10,000 or clip reflectance here:
    TerraMind's published mean/std are expressed in raw digital-number units.
    """

    if source.ndim != 3 or source.shape[0] != 13 or min(source.shape[1:]) < 1:
        raise ValueError("EuroSAT multispectral input must have shape (13, height, width)")
    means = np.asarray(mean, dtype=np.float32)
    stds = np.asarray(std, dtype=np.float32)
    if means.shape != (13,) or stds.shape != (13,):
        raise ValueError("TerraMind normalization requires 13 means and standard deviations")
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError(
            "normalization statistics must be finite with positive standard deviations"
        )
    if not np.isfinite(source).all():
        raise ValueError("multispectral input contains non-finite values")
    ordered = source[np.asarray(BAND_INDICES)].astype(np.float32)
    return (ordered - means[:, None, None]) / stds[:, None, None]


def backbone_state(payload: Any, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Drop unused decoder/modalities but never fill missing backbone weights randomly."""

    if not isinstance(payload, Mapping):
        raise ValueError("TerraMind checkpoint is not a state dictionary")
    missing = sorted(set(expected) - set(payload))
    mismatch = sorted(
        key for key in expected if key in payload and payload[key].shape != expected[key].shape
    )
    if missing or mismatch:
        raise ValueError(
            f"TerraMind backbone mismatch: missing={missing}, shape_mismatch={mismatch}"
        )
    return {key: payload[key] for key in expected}


def terramind_embeddings(
    records: Sequence[ImageRecord],
    *,
    archive: Path,
    checkpoint: Path,
    batch_size: int = 2,
    device: str = "auto",
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Embed the selected existing EuroSAT records; no download or training occurs."""

    if not records or batch_size < 1:
        raise ValueError("records must be non-empty and batch_size must be positive")
    members = archive_members(records)
    verify_archive(archive)
    verify_sha256(checkpoint, CHECKPOINT_SHA256)
    try:
        import torch
        from terratorch import BACKBONE_REGISTRY
        from terratorch.models.backbones.terramind.model.terramind_register import (
            v1_pretraining_mean,
            v1_pretraining_std,
        )
    except ImportError as error:
        raise RuntimeError("TerraMind requires the 'foundation', 'ml', and 'geo' groups") from error

    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected_device == "auto":
        selected_device = "cpu"
    torch.manual_seed(42)
    model = BACKBONE_REGISTRY.build(MODEL_NAME, pretrained=False, modalities=["S2L1C"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(backbone_state(payload, model.state_dict()), strict=True)
    model.requires_grad_(False).eval().to(selected_device)
    del payload
    means = v1_pretraining_mean[MODALITY_KEY]
    stds = v1_pretraining_std[MODALITY_KEY]
    batches = []
    with zipfile.ZipFile(archive) as bundle, torch.inference_mode():
        for start in range(0, len(members), batch_size):
            arrays = [
                prepare_multispectral(read_archive_member(bundle, member), means, stds)
                for member in members[start : start + batch_size]
            ]
            tensor = torch.from_numpy(np.stack(arrays)).to(selected_device)
            tensor = torch.nn.functional.interpolate(
                tensor, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
            )
            layers = model({"S2L1C": tensor})
            patches = layers[-1]
            if tuple(patches.shape) != (len(arrays), 196, 192):
                raise ValueError("unexpected TerraMind-Tiny final-layer patch shape")
            pooled = patches.mean(dim=1)
            if not torch.isfinite(pooled).all() or (pooled.norm(dim=1) == 0).any():
                raise ValueError("TerraMind produced non-finite or zero-length features")
            vectors = torch.nn.functional.normalize(pooled, dim=1)
            batches.append(vectors.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(batches), {
        "backend": "terramind",
        "model": MODEL_NAME,
        "checkpoint_repository": CHECKPOINT_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "archive_md5": EUROSAT_ARCHIVE_MD5,
        "bands": list(BAND_ORDER),
        "normalization_mean": list(means),
        "normalization_std": list(stds),
        "device_actual": selected_device,
        "precision": "float32",
        "frozen": True,
        "pooling": "final-normalized-layer-mean-over-196-patches-then-L2",
        "preprocessing": "raw L1C DN, fixed pretraining standardization, bilinear resize 224",
        "evidence_role": "eurosat-v1-regression-not-independent-confirmation",
    }
