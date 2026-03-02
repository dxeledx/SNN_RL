from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ACTION_CONTINUE = 0
ACTION_COMMIT_LEFT = 1
ACTION_COMMIT_RIGHT = 2


@dataclass(frozen=True)
class StopAndDecideStep:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class StopAndDecideEnv:
    """
    Offline-driven closed-loop env:
      - Each episode samples one trial (sequence of window-features).
      - Actions: CONTINUE / COMMIT_LEFT / COMMIT_RIGHT.
      - Reward: (+1/-1) on commit, and per-step time cost.
    """

    def __init__(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        time_cost: float,
        no_commit_penalty: float = -1.0,
        rng: np.random.Generator,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(f"Expected X [N,S,D], got {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"Expected y [N] matching X, got y={y.shape} X={X.shape}")
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.time_cost = float(time_cost)
        self.no_commit_penalty = float(no_commit_penalty)
        self.rng = rng

        self.n_trials, self.n_steps, self.obs_dim = self.X.shape
        self._trial_idx: int | None = None
        self._step_idx: int = 0

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self._trial_idx = int(self.rng.integers(0, self.n_trials))
        self._step_idx = 0
        obs = self.X[self._trial_idx, self._step_idx]
        info = {"trial_idx": self._trial_idx, "t": self._step_idx, "label": int(self.y[self._trial_idx])}
        return obs.copy(), info

    def step(self, action: int) -> StopAndDecideStep:
        if self._trial_idx is None:
            raise RuntimeError("Call reset() before step().")
        action = int(action)
        label = int(self.y[self._trial_idx])
        terminated = False
        committed = False

        reward = -self.time_cost
        if action == ACTION_CONTINUE:
            if self._step_idx >= (self.n_steps - 1):
                terminated = True
                reward += self.no_commit_penalty
            else:
                self._step_idx += 1
        elif action in (ACTION_COMMIT_LEFT, ACTION_COMMIT_RIGHT):
            terminated = True
            committed = True
            pred = 0 if action == ACTION_COMMIT_LEFT else 1
            reward += 1.0 if pred == label else -1.0
        else:
            raise ValueError(f"Invalid action: {action}")

        if terminated:
            obs = np.zeros((self.obs_dim,), dtype=np.float32)
        else:
            obs = self.X[self._trial_idx, self._step_idx]

        info = {
            "trial_idx": int(self._trial_idx),
            "t": int(self._step_idx),
            "label": label,
            "committed": committed,
            "done": terminated,
        }
        return StopAndDecideStep(obs=obs.copy(), reward=float(reward), done=bool(terminated), info=info)

