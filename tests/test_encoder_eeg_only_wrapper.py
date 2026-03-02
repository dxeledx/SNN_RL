from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bci_snn_rl.models.encoders import EegOnlyWrapperEncoder, SigmaDeltaConfig, SigmaDeltaEncoder  # noqa: E402


def test_eeg_only_wrapper_keeps_aux_features_unchanged() -> None:
    enc = SigmaDeltaEncoder(cfg=SigmaDeltaConfig(learnable_threshold=False, theta_init=0.25))
    wrapped = EegOnlyWrapperEncoder(base_encoder=enc, aux_dim=3)

    # main features should be encoded, aux features should pass through.
    x_main = torch.ones((2, 4), dtype=torch.float32)
    x_aux = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=torch.float32)
    x = torch.cat([x_main, x_aux], dim=-1)

    y = wrapped(x)
    assert tuple(y.shape) == tuple(x.shape)
    assert torch.allclose(y[:, -3:], x_aux)
    assert torch.any(y[:, :-3] != 0.0)

