from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBuffer:
    obs: torch.Tensor  # [T, N, D]
    actions: torch.Tensor  # [T, N]
    logprobs: torch.Tensor  # [T, N]
    rewards: torch.Tensor  # [T, N]
    dones: torch.Tensor  # [T, N] bool
    values: torch.Tensor  # [T, N]

    advantages: torch.Tensor  # [T, N]
    returns: torch.Tensor  # [T, N]

    t: int = 0

    @classmethod
    def allocate(cls, *, T: int, N: int, obs_dim: int, device: torch.device) -> "RolloutBuffer":
        zeros_f = lambda *shape: torch.zeros(shape, device=device, dtype=torch.float32)  # noqa: E731
        zeros_b = lambda *shape: torch.zeros(shape, device=device, dtype=torch.bool)  # noqa: E731
        return cls(
            obs=zeros_f(T, N, obs_dim),
            actions=torch.zeros((T, N), device=device, dtype=torch.int64),
            logprobs=zeros_f(T, N),
            rewards=zeros_f(T, N),
            dones=zeros_b(T, N),
            values=zeros_f(T, N),
            advantages=zeros_f(T, N),
            returns=zeros_f(T, N),
            t=0,
        )

    def add(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        if self.t >= self.obs.shape[0]:
            raise RuntimeError("RolloutBuffer overflow")
        self.obs[self.t].copy_(obs)
        self.actions[self.t].copy_(actions)
        self.logprobs[self.t].copy_(logprobs)
        self.rewards[self.t].copy_(rewards)
        self.dones[self.t].copy_(dones)
        self.values[self.t].copy_(values)
        self.t += 1

    def compute_returns_advantages(self, *, last_values: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        T, _N = self.rewards.shape
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros((_N,), device=self.rewards.device, dtype=torch.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
                next_non_terminal = (~self.dones[t]).float()
            else:
                next_values = self.values[t + 1]
                next_non_terminal = (~self.dones[t]).float()

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = self.advantages + self.values

    def get(self) -> tuple[torch.Tensor, ...]:
        # Flatten T and N
        T, N, D = self.obs.shape
        b_obs = self.obs.reshape(T * N, D)
        b_actions = self.actions.reshape(T * N)
        b_logprobs = self.logprobs.reshape(T * N)
        b_advantages = self.advantages.reshape(T * N)
        b_returns = self.returns.reshape(T * N)
        b_values = self.values.reshape(T * N)
        return b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values


@dataclass
class RolloutBufferContinuous:
    obs: torch.Tensor  # [T, N, D]
    actions: torch.Tensor  # [T, N, A]
    logprobs: torch.Tensor  # [T, N]
    rewards: torch.Tensor  # [T, N]
    dones: torch.Tensor  # [T, N] bool
    values: torch.Tensor  # [T, N]

    advantages: torch.Tensor  # [T, N]
    returns: torch.Tensor  # [T, N]

    t: int = 0

    @classmethod
    def allocate(cls, *, T: int, N: int, obs_dim: int, action_dim: int, device: torch.device) -> "RolloutBufferContinuous":
        zeros_f = lambda *shape: torch.zeros(shape, device=device, dtype=torch.float32)  # noqa: E731
        zeros_b = lambda *shape: torch.zeros(shape, device=device, dtype=torch.bool)  # noqa: E731
        return cls(
            obs=zeros_f(T, N, obs_dim),
            actions=zeros_f(T, N, action_dim),
            logprobs=zeros_f(T, N),
            rewards=zeros_f(T, N),
            dones=zeros_b(T, N),
            values=zeros_f(T, N),
            advantages=zeros_f(T, N),
            returns=zeros_f(T, N),
            t=0,
        )

    def add(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        if self.t >= self.obs.shape[0]:
            raise RuntimeError("RolloutBuffer overflow")
        self.obs[self.t].copy_(obs)
        self.actions[self.t].copy_(actions)
        self.logprobs[self.t].copy_(logprobs)
        self.rewards[self.t].copy_(rewards)
        self.dones[self.t].copy_(dones)
        self.values[self.t].copy_(values)
        self.t += 1

    def compute_returns_advantages(self, *, last_values: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        T, _N = self.rewards.shape
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros((_N,), device=self.rewards.device, dtype=torch.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
                next_non_terminal = (~self.dones[t]).float()
            else:
                next_values = self.values[t + 1]
                next_non_terminal = (~self.dones[t]).float()

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = self.advantages + self.values
