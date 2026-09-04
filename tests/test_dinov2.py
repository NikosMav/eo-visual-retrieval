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

from eo_visual_retrieval.embeddings import dinov2 as dinov2_module
from eo_visual_retrieval.embeddings.dinov2 import (
    ALLOWED_MODELS,
    HUB_REF,
    HUB_REPO,
    HUB_TREE_SHA256,
    _hub_source_root,
    _resolve_device,
    _verify_hub_source,
    dinov2_embeddings,
    hub_provenance,
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
    # The stub replaces the download, so there is no extracted tree to verify.
    # The verification itself is covered by the tests below.
    monkeypatch.setattr(dinov2_module, "_verify_hub_source", lambda torch: None)

    paths = [_image(tmp_path, f"chip-{index}.png") for index in range(3)]
    vectors = dinov2_embeddings(paths, batch_size=2, device="cpu")

    assert vectors.shape == (3, 4)
    assert vectors.dtype == np.float32
    # Non-square inputs are resized to the documented 224x224 model geometry.
    assert seen == [(2, 3, 224, 224), (1, 3, 224, 224)]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6, atol=1e-6)


def _hub(root: Path) -> SimpleNamespace:
    return SimpleNamespace(hub=SimpleNamespace(get_dir=lambda: str(root)))


def _tree(root: Path, contents: dict[str, str]) -> Path:
    for name, text in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def test_the_hub_reference_is_an_immutable_commit() -> None:
    """A branch name would resolve to different code on a later run."""

    assert HUB_REPO == "facebookresearch/dinov2"
    assert len(HUB_REF) == 40
    assert set(HUB_REF) <= set("0123456789abcdef")
    assert hub_provenance() == {
        "hub_repo": HUB_REPO,
        "hub_ref": HUB_REF,
        "hub_tree_sha256": HUB_TREE_SHA256,
    }


def test_the_pinned_reference_is_requested_and_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    requested: list[str] = []
    verified: list[bool] = []

    class StubModel:
        def eval(self) -> StubModel:
            return self

        def to(self, device: str) -> StubModel:
            return self

        def __call__(self, tensor: Any) -> Any:
            return torch.full((tensor.shape[0], 4), 3.0)

    def record(github: str, *args: Any, **kwargs: Any) -> StubModel:
        requested.append(github)
        return StubModel()

    monkeypatch.setattr(torch.hub, "load", record)
    monkeypatch.setattr(
        dinov2_module, "_verify_hub_source", lambda torch: verified.append(True)
    )

    dinov2_embeddings([_image(tmp_path)], device="cpu")

    assert requested == [f"{HUB_REPO}:{HUB_REF}"]
    assert verified == [True], "the delivered code must be checked, not only requested"


def test_the_source_root_is_named_for_the_commit(tmp_path: Path) -> None:
    root = _hub_source_root(_hub(tmp_path))

    assert root == tmp_path / f"facebookresearch_dinov2_{HUB_REF}"


def test_missing_pinned_source_is_reported_rather_than_assumed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="cannot verify pinned DINOv2 code"):
        _verify_hub_source(_hub(tmp_path))


def test_code_that_is_not_the_pinned_commit_is_refused(tmp_path: Path) -> None:
    """Requesting a commit does not prove the bytes that arrived are that commit."""

    _tree(tmp_path / f"facebookresearch_dinov2_{HUB_REF}", {"hubconf.py": "raise SystemExit\n"})

    with pytest.raises(RuntimeError, match="refusing to embed with model code"):
        _verify_hub_source(_hub(tmp_path))


def test_verification_ignores_bytecode_written_by_the_first_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second run imports the same code and must not hash a different tree."""

    from eo_visual_retrieval.hashing import source_tree_sha256

    root = _tree(tmp_path / f"facebookresearch_dinov2_{HUB_REF}", {"hubconf.py": "x = 1\n"})
    monkeypatch.setattr(
        dinov2_module,
        "HUB_TREE_SHA256",
        source_tree_sha256(root, skip_directories=frozenset({"__pycache__"})),
    )
    _verify_hub_source(_hub(tmp_path))

    _tree(root, {"__pycache__/hubconf.cpython-312.pyc": "compiled\n"})
    _verify_hub_source(_hub(tmp_path))
