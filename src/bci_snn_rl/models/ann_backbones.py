from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ANNMLPConfig:
    hidden_sizes: list[int]


class ANNMLP(nn.Module):
    def __init__(self, *, in_dim: int, cfg: ANNMLPConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        cur = int(in_dim)
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(cur, int(h)))
            layers.append(nn.Tanh())
            cur = int(h)
        self.net = nn.Sequential(*layers)
        self.spike_rate = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.spike_rate = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return self.net(x)

