from __future__ import annotations

import numpy as np
import pytest

from eo_visual_retrieval.embeddings.terramind import (
    BAND_ORDER,
    backbone_state,
    prepare_multispectral,
)


def test_terramind_uses_raw_dn_and_reorders_b8a() -> None:
    source = np.stack([np.full((2, 2), 1000 * (i + 1)) for i in range(13)])
    result = prepare_multispectral(source, [100] * 13, [10] * 13)
    assert BAND_ORDER[8] == "B8A"
    assert result.dtype == np.float32
    assert result[:, 0, 0].tolist() == pytest.approx(
        [90, 190, 290, 390, 490, 590, 690, 790, 1290, 890, 990, 1090, 1190]
    )


@pytest.mark.parametrize("shape", [(12, 2, 2), (13, 0, 2), (13, 2)])
def test_terramind_rejects_invalid_tensor(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        prepare_multispectral(np.zeros(shape), [0] * 13, [1] * 13)


def test_terramind_rejects_invalid_statistics() -> None:
    with pytest.raises(ValueError, match="positive"):
        prepare_multispectral(np.zeros((13, 2, 2)), [0] * 13, [0] * 13)
    with pytest.raises(ValueError, match="13 means"):
        prepare_multispectral(np.zeros((13, 2, 2)), [0] * 12, [1] * 13)
    with pytest.raises(ValueError, match="non-finite"):
        prepare_multispectral(np.full((13, 2, 2), np.nan), [0] * 13, [1] * 13)


def test_backbone_state_drops_unused_decoder_but_never_keeps_random_weights() -> None:
    expected = {"encoder.weight": np.zeros((2, 3))}
    payload = {"encoder.weight": np.ones((2, 3)), "decoder.weight": np.ones((4, 5))}
    state = backbone_state(payload, expected)
    assert list(state) == ["encoder.weight"]
    assert np.all(state["encoder.weight"] == 1)
    with pytest.raises(ValueError, match="missing"):
        backbone_state({}, expected)
    with pytest.raises(ValueError, match="shape_mismatch"):
        backbone_state({"encoder.weight": np.ones((4, 3))}, expected)
    with pytest.raises(ValueError, match="state dictionary"):
        backbone_state([], expected)
