from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eo_visual_retrieval.embeddings.ssl4eo import (
    SSL4EO_BAND_INDICES,
    SSL4EO_BAND_ORDER,
    _checkpoint_state,
    prepare_multispectral,
    verify_sha256,
)


def test_prepare_multispectral_reorders_b8a_and_scales_digital_numbers() -> None:
    source = np.stack(
        [np.full((2, 3), (index + 1) * 1_000, dtype=np.uint16) for index in range(13)]
    )
    source[11] = 20_000

    prepared = prepare_multispectral(source)

    assert prepared.shape == (13, 2, 3)
    assert prepared.dtype == np.float32
    assert SSL4EO_BAND_ORDER[8] == "B8A"
    assert SSL4EO_BAND_INDICES[8] == 12
    assert prepared[:, 0, 0].tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 0.9, 1.0, 1.0, 1.0]
    )


@pytest.mark.parametrize("shape", [(12, 2, 2), (13, 2), (2, 13, 2, 2)])
def test_prepare_multispectral_rejects_wrong_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"shape \(13, height, width\)"):
        prepare_multispectral(np.zeros(shape, dtype=np.uint16))


def test_verify_sha256_accepts_expected_digest_and_rejects_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    expected = "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"

    assert verify_sha256(checkpoint, expected) == expected
    with pytest.raises(ValueError, match="checkpoint checksum mismatch"):
        verify_sha256(checkpoint, "0" * 64)


def test_checkpoint_state_extracts_only_original_moco_query_encoder() -> None:
    payload = {
        "state_dict": {
            "module.encoder_q.conv1.weight": "query-convolution",
            "module.encoder_q.fc.0.weight": "projection-head",
            "module.encoder_k.conv1.weight": "key-convolution",
            "module.queue": "queue",
        }
    }

    assert _checkpoint_state(payload) == {"conv1.weight": "query-convolution"}
