from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


def _require_spikingjelly() -> None:
    try:
        import spikingjelly  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("SpikingJelly is required for SNN actor. Install `spikingjelly`.") from e


@dataclass(frozen=True)
class SpikingMLPConfig:
    hidden_sizes: list[int]
    input_scale: float = 1.0
    v_threshold: float = 1.0
    # "spike": return spikes of last LIF layer (0/1)
    # "membrane": return membrane potential v of last LIF layer (continuous, typically improves decoding accuracy)
    output_mode: Literal["spike", "membrane"] = "spike"


class SpikingMLP(nn.Module):
    def __init__(self, *, in_dim: int, cfg: SpikingMLPConfig) -> None:
        super().__init__()
        _require_spikingjelly()
        from spikingjelly.activation_based import neuron, surrogate

        self.input_scale = float(cfg.input_scale)
        self.output_mode = str(cfg.output_mode)
        layers: list[nn.Module] = []
        cur = int(in_dim)
        self._lif_nodes: list[nn.Module] = []
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(cur, int(h)))
            lif = neuron.LIFNode(
                v_threshold=float(cfg.v_threshold),
                surrogate_function=surrogate.ATan(),
                detach_reset=True,
            )
            layers.append(lif)
            self._lif_nodes.append(lif)
            cur = int(h)

        self.net = nn.Sequential(*layers)
        self.spike_rate: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * float(self.input_scale)
        spike_sum = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        spike_count = 0
        last_mem: torch.Tensor | None = None

        for layer in self.net:
            x = layer(x)
            if layer in self._lif_nodes:
                # output of LIFNode is spikes (0/1). Treat as differentiable proxy for rate.
                spike_sum = spike_sum + x.abs().mean()
                spike_count += 1
                # SpikingJelly stores membrane potential in `v`.
                v = getattr(layer, "v", None)
                if isinstance(v, torch.Tensor):
                    last_mem = v

        if spike_count > 0:
            self.spike_rate = spike_sum / float(spike_count)
        else:
            self.spike_rate = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.output_mode == "membrane" and last_mem is not None:
            return last_mem
        return x
