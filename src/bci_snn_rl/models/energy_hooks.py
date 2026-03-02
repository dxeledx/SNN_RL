from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SpikeCounter:
    spikes: float = 0.0
    elements: float = 0.0

    def reset(self) -> None:
        self.spikes = 0.0
        self.elements = 0.0

    def update(self, x: torch.Tensor) -> None:
        self.spikes += float(x.detach().abs().sum().item())
        self.elements += float(x.numel())

    def rate(self) -> float:
        if self.elements <= 0:
            return 0.0
        return float(self.spikes / self.elements)


def attach_spike_counter(module: nn.Module) -> SpikeCounter:
    """
    Attach forward hooks to count spikes (by abs(output)).
    Intended for evaluation/diagnostics only.
    """
    counter = SpikeCounter()

    def _hook(_m: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        if isinstance(out, torch.Tensor):
            counter.update(out)

    # Heuristic: only attach to modules with `v` attribute (SpikingJelly neurons).
    for m in module.modules():
        if hasattr(m, "v"):
            m.register_forward_hook(_hook)

    return counter

