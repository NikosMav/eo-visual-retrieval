from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eo_visual_retrieval.embeddings.ssl4eo import (
    SSL4EO_ALL,
    SSL4EO_BAND_INDICES,
    SSL4EO_BAND_ORDER,
    SSL4EO_RGB,
    _checkpoint_state,
    prepare_bands,
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


def test_rgb_variant_selects_red_green_blue_in_torchgeo_order() -> None:
    """torchgeo registers this checkpoint's bands as ['B4', 'B3', 'B2'] — red first."""
    source = np.stack(
        [np.full((2, 3), (index + 1) * 1_000, dtype=np.uint16) for index in range(13)]
    )

    prepared = prepare_bands(source, SSL4EO_RGB.band_indices)

    assert SSL4EO_RGB.band_order == ("B04", "B03", "B02")
    assert prepared.shape == (3, 2, 3)
    # EUROSAT_BAND_ORDER positions: B04 is index 3, B03 is 2, B02 is 1.
    assert prepared[:, 0, 0].tolist() == pytest.approx([0.4, 0.3, 0.2])


def test_both_variants_share_one_scaling_so_only_bands_differ() -> None:
    """The ablation is only controlled if preprocessing is byte-identical."""
    source = np.stack(
        [np.full((2, 2), (index + 1) * 1_000, dtype=np.uint16) for index in range(13)]
    )

    thirteen = prepare_bands(source, SSL4EO_ALL.band_indices)
    rgb = prepare_bands(source, SSL4EO_RGB.band_indices)

    red_in_thirteen = SSL4EO_ALL.band_order.index("B04")
    np.testing.assert_array_equal(rgb[0], thirteen[red_in_thirteen])
    assert rgb.dtype == thirteen.dtype == np.float32


def test_rgb_variant_pins_the_published_checkpoint_identity() -> None:
    assert SSL4EO_RGB.checkpoint_repository == "torchgeo/resnet50_sentinel2_rgb_moco"
    assert SSL4EO_RGB.checkpoint_filename == "resnet50_sentinel2_rgb_moco-2b57ba8b.pth"
    assert SSL4EO_RGB.checkpoint_sha256 == (
        "2b57ba8b9964dbe1c409aac1bb79b4d97c19c874ffe7934799b7c8ad94ff85f0"
    )
    # torchgeo names these files by their content digest.
    assert SSL4EO_RGB.checkpoint_filename.endswith(
        f"-{SSL4EO_RGB.checkpoint_sha256[:8]}.pth"
    )
    assert SSL4EO_ALL.checkpoint_filename.endswith(
        f"-{SSL4EO_ALL.checkpoint_sha256[:8]}.pth"
    )


def test_variant_channel_counts_match_their_band_orders() -> None:
    assert SSL4EO_ALL.channels == 13
    assert SSL4EO_RGB.channels == 3


def test_prepare_bands_rejects_input_that_is_not_the_full_archive_stack() -> None:
    with pytest.raises(ValueError, match=r"shape \(13, height, width\)"):
        prepare_bands(np.zeros((3, 2, 2), dtype=np.uint16), SSL4EO_RGB.band_indices)
