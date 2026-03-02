from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from bci_snn_rl.models.encoders import (
    DeltaEncoder,
    EegOnlyWrapperEncoder,
    IdentityEncoder,
    SigmaDeltaConfig,
    SigmaDeltaEncoder,
)
from bci_snn_rl.models.snn_backbones import SpikingMLP, SpikingMLPConfig
from bci_snn_rl.models.ann_backbones import ANNMLP, ANNMLPConfig


@dataclass(frozen=True)
class ActorCriticConfig:
    encoder_type: Literal["sigma_delta", "delta", "none"]
    encoder_learnable_threshold: bool
    encoder_theta_init: float
    actor_type: Literal["snn_mlp", "ann_mlp"]
    actor_hidden_sizes: list[int]
    critic_hidden_sizes: list[int]
    # SNN-only knobs (safe to ignore for ANN actor).
    snn_input_scale: float = 1.0
    snn_v_threshold: float = 1.0
    snn_output_mode: Literal["spike", "membrane"] = "spike"
    encoder_apply_to: Literal["all", "eeg_only"] = "all"
    encoder_aux_dim: int = 0


def _make_encoder(cfg: ActorCriticConfig) -> nn.Module:
    if cfg.encoder_type == "none":
        return IdentityEncoder()

    if cfg.encoder_type == "delta":
        base: nn.Module = DeltaEncoder()
    elif cfg.encoder_type == "sigma_delta":
        base = SigmaDeltaEncoder(
            cfg=SigmaDeltaConfig(learnable_threshold=cfg.encoder_learnable_threshold, theta_init=cfg.encoder_theta_init)
        )
    else:
        raise ValueError(f"Unknown encoder.type: {cfg.encoder_type}")

    apply_to = str(cfg.encoder_apply_to)
    if apply_to not in ("all", "eeg_only"):
        raise ValueError(f"Unknown encoder.apply_to: {apply_to}")
    if apply_to == "eeg_only" and int(cfg.encoder_aux_dim) > 0:
        return EegOnlyWrapperEncoder(base_encoder=base, aux_dim=int(cfg.encoder_aux_dim))
    return base


class ActorCritic(nn.Module):
    def __init__(self, *, obs_dim: int, action_dim: int, cfg: ActorCriticConfig) -> None:
        super().__init__()
        self.encoder = _make_encoder(cfg)
        if cfg.actor_type == "snn_mlp":
            self.actor_backbone: nn.Module = SpikingMLP(
                in_dim=obs_dim,
                cfg=SpikingMLPConfig(
                    hidden_sizes=cfg.actor_hidden_sizes,
                    input_scale=float(cfg.snn_input_scale),
                    v_threshold=float(cfg.snn_v_threshold),
                    output_mode=str(cfg.snn_output_mode),
                ),
            )
            actor_out_dim = int(cfg.actor_hidden_sizes[-1])
        elif cfg.actor_type == "ann_mlp":
            self.actor_backbone = ANNMLP(in_dim=obs_dim, cfg=ANNMLPConfig(hidden_sizes=cfg.actor_hidden_sizes))
            actor_out_dim = int(cfg.actor_hidden_sizes[-1])
        else:
            raise ValueError(f"Unknown model.actor.type: {cfg.actor_type}")
        self.actor_head = nn.Linear(actor_out_dim, int(action_dim))

        critic_layers: list[nn.Module] = []
        cur = int(obs_dim)
        for h in cfg.critic_hidden_sizes:
            critic_layers.append(nn.Linear(cur, int(h)))
            critic_layers.append(nn.Tanh())
            cur = int(h)
        critic_layers.append(nn.Linear(cur, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.last_spike_rate: torch.Tensor = torch.tensor(0.0)

    def forward(self, obs: torch.Tensor, *, reset_state: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if reset_state:
            self._reset_spiking_state_all(self.actor_backbone)
        x = self.encoder(obs)
        h = self.actor_backbone(x)
        logits = self.actor_head(h)
        value = self.critic(obs).squeeze(-1)
        sr = getattr(self.actor_backbone, "spike_rate", None)
        self.last_spike_rate = sr if isinstance(sr, torch.Tensor) else torch.tensor(0.0, device=obs.device)
        return logits, value, self.last_spike_rate

    @torch.no_grad()
    def reset_done(self, done_mask: torch.Tensor) -> None:
        done_mask = done_mask.to(dtype=torch.bool)
        self._reset_spiking_state_by_mask(self.actor_backbone, done_mask)
        if hasattr(self.encoder, "reset_mask"):
            self.encoder.reset_mask(done_mask)

    @torch.no_grad()
    def reset_all_states(self) -> None:
        self._reset_spiking_state_all(self.actor_backbone)
        self._reset_encoder_state(self.encoder, device=next(self.parameters()).device)

    @staticmethod
    def _reset_encoder_state(encoder: nn.Module, *, device: torch.device) -> None:
        # Clear encoder internal state (including wrapped encoders) so the next forward re-initializes from input shape.
        for m in encoder.modules():
            for attr in ("_prev", "_acc"):
                if hasattr(m, attr):
                    try:
                        setattr(m, attr, torch.tensor([], device=device))
                    except Exception:
                        setattr(m, attr, torch.tensor([]))

    @staticmethod
    def _reset_spiking_state_by_mask(module: nn.Module, done_mask: torch.Tensor) -> None:
        # SpikingJelly activation_based neurons store membrane potential in `v`.
        for m in module.modules():
            v = getattr(m, "v", None)
            if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == done_mask.shape[0]:
                keep = (~done_mask).to(dtype=v.dtype, device=v.device)
                view = (keep.shape[0],) + (1,) * (v.ndim - 1)
                # IMPORTANT: avoid in-place mutation of `v` which can break autograd.
                setattr(m, "v", v.detach() * keep.view(view))

    @staticmethod
    def _reset_spiking_state_all(module: nn.Module) -> None:
        # Make spiking state shape-agnostic (important for PPO minibatch updates).
        for m in module.modules():
            if hasattr(m, "v"):
                try:
                    setattr(m, "v", 0.0)
                except Exception:
                    pass


class GaussianActorCritic(nn.Module):
    def __init__(self, *, obs_dim: int, action_dim: int, cfg: ActorCriticConfig) -> None:
        super().__init__()
        self.encoder = _make_encoder(cfg)

        if cfg.actor_type == "snn_mlp":
            self.actor_backbone: nn.Module = SpikingMLP(
                in_dim=obs_dim,
                cfg=SpikingMLPConfig(
                    hidden_sizes=cfg.actor_hidden_sizes,
                    input_scale=float(cfg.snn_input_scale),
                    v_threshold=float(cfg.snn_v_threshold),
                    output_mode=str(cfg.snn_output_mode),
                ),
            )
            actor_out_dim = int(cfg.actor_hidden_sizes[-1])
        elif cfg.actor_type == "ann_mlp":
            self.actor_backbone = ANNMLP(in_dim=obs_dim, cfg=ANNMLPConfig(hidden_sizes=cfg.actor_hidden_sizes))
            actor_out_dim = int(cfg.actor_hidden_sizes[-1])
        else:
            raise ValueError(f"Unknown model.actor.type: {cfg.actor_type}")

        self.actor_mean = nn.Linear(actor_out_dim, int(action_dim))
        self.actor_log_std = nn.Parameter(torch.zeros((int(action_dim),), dtype=torch.float32))

        critic_layers: list[nn.Module] = []
        cur = int(obs_dim)
        for h in cfg.critic_hidden_sizes:
            critic_layers.append(nn.Linear(cur, int(h)))
            critic_layers.append(nn.Tanh())
            cur = int(h)
        critic_layers.append(nn.Linear(cur, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.last_spike_rate: torch.Tensor = torch.tensor(0.0)

    def forward(
        self, obs: torch.Tensor, *, reset_state: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if reset_state:
            self._reset_spiking_state_all(self.actor_backbone)
        x = self.encoder(obs)
        h = self.actor_backbone(x)
        mean = self.actor_mean(h)
        log_std = self.actor_log_std.expand_as(mean)
        value = self.critic(obs).squeeze(-1)
        sr = getattr(self.actor_backbone, "spike_rate", None)
        self.last_spike_rate = sr if isinstance(sr, torch.Tensor) else torch.tensor(0.0, device=obs.device)
        return mean, log_std, value, self.last_spike_rate

    @torch.no_grad()
    def reset_done(self, done_mask: torch.Tensor) -> None:
        done_mask = done_mask.to(dtype=torch.bool)
        self._reset_spiking_state_by_mask(self.actor_backbone, done_mask)
        if hasattr(self.encoder, "reset_mask"):
            self.encoder.reset_mask(done_mask)

    @torch.no_grad()
    def reset_all_states(self) -> None:
        self._reset_spiking_state_all(self.actor_backbone)
        ActorCritic._reset_encoder_state(self.encoder, device=next(self.parameters()).device)

    @staticmethod
    def _reset_spiking_state_by_mask(module: nn.Module, done_mask: torch.Tensor) -> None:
        for m in module.modules():
            v = getattr(m, "v", None)
            if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == done_mask.shape[0]:
                keep = (~done_mask).to(dtype=v.dtype, device=v.device)
                view = (keep.shape[0],) + (1,) * (v.ndim - 1)
                setattr(m, "v", v.detach() * keep.view(view))

    @staticmethod
    def _reset_spiking_state_all(module: nn.Module) -> None:
        for m in module.modules():
            if hasattr(m, "v"):
                try:
                    setattr(m, "v", 0.0)
                except Exception:
                    pass
