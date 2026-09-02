"""DINOv2 guards and preprocessing, with the checkpoint stubbed out.

The guards run before any network access, so they are exercised everywhere. The
full path needs torch and is skipped in the lightweight CI environment.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from eo_visual_retrieval.embeddings.dinov2 import (
    ALLOWED_MODELS,
    _resolve_device,
    dinov2_embeddings,
)


def _image(tmp_path: Path, name: str = "chip.png") -> Path:
    from PIL import Image

    path = tmp_path / name
    Image.fromarray(np.full((12, 20, 3), 90, dtype=np.uint8)).save(path)
    return path


def test_only_validated_checkpoints_are_accepted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported DINOv2 model"):
        dinov2_embeddings([_image(tmp_path)], model_name="resnet50")
    assert "dinov2_vits14" in ALLOWED_MODELS


def test_invalid_batch_and_empty_input_fail_before_the_checkpoint_is_fetched(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        dinov2_embeddings([_image(tmp_path)], batch_size=0)
    with pytest.raises(ValueError, match="at least one image is required"):
        dinov2_embeddings([])


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [("cpu", True, "cpu"), ("cuda", False, "cuda"), ("auto", True, "cuda"), ("auto", False, "cpu")],
)
def test_auto_device_follows_cuda_availability(
    requested: str, available: bool, expected: str
) -> None:
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available))

    assert _resolve_device(torch, requested) == expected


def test_preprocessing_and_batching_with_a_stubbed_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    seen: list[tuple[int, ...]] = []

    class StubModel:
        def eval(self) -> StubModel:
            return self

        def to(self, device: str) -> StubModel:
            return self

        def __call__(self, tensor: Any) -> Any:
            seen.append(tuple(tensor.shape))
            # A constant non-unit response proves the caller normalizes the output.
            return torch.full((tensor.shape[0], 4), 3.0)

    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: StubModel())

    paths = [_image(tmp_path, f"chip-{index}.png") for index in range(3)]
    vectors = dinov2_embeddings(paths, batch_size=2, device="cpu")

    assert vectors.shape == (3, 4)
    assert vectors.dtype == np.float32
    # Non-square inputs are resized to the documented 224x224 model geometry.
    assert seen == [(2, 3, 224, 224), (1, 3, 224, 224)]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6, atol=1e-6)
