from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class IdentityEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x

    def reset_mask(self, done_mask: torch.Tensor) -> None:  # noqa: ARG002
        return


class DeltaEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_prev", torch.tensor([]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._prev.numel() == 0 or self._prev.shape != x.shape:
            self._prev = torch.zeros_like(x)
        out = x - self._prev
        self._prev = x.detach()
        return out

    def reset_mask(self, done_mask: torch.Tensor) -> None:
        if self._prev.numel() == 0:
            return
        if self._prev.shape[0] != done_mask.shape[0]:
            return
        done_mask = done_mask.to(dtype=torch.bool, device=self._prev.device)
        keep = (~done_mask).to(dtype=self._prev.dtype)
        view = (keep.shape[0],) + (1,) * (self._prev.ndim - 1)
        self._prev = self._prev.detach() * keep.view(view)


@dataclass(frozen=True)
class SigmaDeltaConfig:
    learnable_threshold: bool
    theta_init: float


class SigmaDeltaEncoder(nn.Module):
    """
    Feature-level Sigma-Delta encoder:
      acc <- acc + (x - prev)
      emit +1/-1 when |acc| >= theta, then acc <- acc - emit*theta
    """

    def __init__(self, *, cfg: SigmaDeltaConfig) -> None:
        super().__init__()
        theta_init = float(cfg.theta_init)
        if cfg.learnable_threshold:
            self.theta_raw = nn.Parameter(torch.tensor(theta_init).log())
        else:
            self.register_buffer("theta_raw", torch.tensor(theta_init).log(), persistent=False)

        self.register_buffer("_prev", torch.tensor([]), persistent=False)
        self.register_buffer("_acc", torch.tensor([]), persistent=False)

    @property
    def theta(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.theta_raw) + 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._prev.numel() == 0 or self._prev.shape != x.shape:
            self._prev = torch.zeros_like(x)
            self._acc = torch.zeros_like(x)

        delta = x - self._prev
        self._prev = x.detach()
        self._acc = self._acc + delta

        theta = self.theta
        emit = torch.zeros_like(self._acc)
        emit = emit + (self._acc >= theta).to(x.dtype)
        emit = emit - (self._acc <= -theta).to(x.dtype)

        self._acc = self._acc - emit * theta
        return emit

    def reset_mask(self, done_mask: torch.Tensor) -> None:
        if self._prev.numel() == 0:
            return
        if self._prev.shape[0] != done_mask.shape[0]:
            return
        done_mask = done_mask.to(dtype=torch.bool, device=self._prev.device)
        keep = (~done_mask).to(dtype=self._prev.dtype)
        view = (keep.shape[0],) + (1,) * (self._prev.ndim - 1)
        self._prev = self._prev.detach() * keep.view(view)
        self._acc = self._acc.detach() * keep.view(view)


class EegOnlyWrapperEncoder(nn.Module):
    """
    Apply an encoder only to the "EEG feature" part of the observation.

    Convention used by this repo's Stop task:
      obs = [eeg_features..., subject_onehot..., t_norm]

    If `aux_dim>0`, the last `aux_dim` features are treated as auxiliary (subject/time)
    and are passed through unchanged.
    """

    def __init__(self, *, base_encoder: nn.Module, aux_dim: int) -> None:
        super().__init__()
        if int(aux_dim) <= 0:
            raise ValueError(f"aux_dim must be > 0, got {aux_dim}")
        self.base_encoder = base_encoder
        self.aux_dim = int(aux_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"Expected x with shape [N,...,D], got {tuple(x.shape)}")
        if int(x.shape[-1]) <= self.aux_dim:
            raise ValueError(f"aux_dim={self.aux_dim} must be < obs_dim={int(x.shape[-1])}")
        main = x[..., :-self.aux_dim]
        aux = x[..., -self.aux_dim :]
        enc = self.base_encoder(main)
        return torch.cat([enc, aux], dim=-1)

    def reset_mask(self, done_mask: torch.Tensor) -> None:  # noqa: D401
        if hasattr(self.base_encoder, "reset_mask"):
            self.base_encoder.reset_mask(done_mask)
