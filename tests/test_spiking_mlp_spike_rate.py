from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly")

from torch import nn  # noqa: E402

from bci_snn_rl.models.snn_backbones import SpikingMLP, SpikingMLPConfig  # noqa: E402


def test_spiking_mlp_can_spike_with_low_threshold() -> None:
    torch.manual_seed(0)

    mlp = SpikingMLP(in_dim=4, cfg=SpikingMLPConfig(hidden_sizes=[8], input_scale=1.0, v_threshold=0.5))
    # Force a strong positive current into the LIF.
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.zero_()
                m.bias.fill_(2.0)

    x = torch.zeros((2, 4), dtype=torch.float32)
    _ = mlp(x)
    assert float(mlp.spike_rate.detach().cpu().item()) > 0.0

