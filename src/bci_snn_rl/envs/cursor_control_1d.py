from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CursorStep:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class CursorControl1DEnv:
    """
    Offline-driven continuous control env:
      - target is -1 for class 0 (left), +1 for class 1 (right)
      - action is velocity in [-1, 1]
    """

    def __init__(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        time_cost: float,
        success_tol: float = 0.1,
        max_abs_pos: float = 2.0,
        action_scale: float = 0.2,
        success_bonus: float = 1.0,
        rng: np.random.Generator,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(f"Expected X [N,S,D], got {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"Expected y [N] matching X, got y={y.shape} X={X.shape}")
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.time_cost = float(time_cost)
        self.success_tol = float(success_tol)
        self.max_abs_pos = float(max_abs_pos)
        self.action_scale = float(action_scale)
        self.success_bonus = float(success_bonus)
        self.rng = rng

        self.n_trials, self.n_steps, self.obs_dim = self.X.shape
        self._trial_idx: int | None = None
        self._step_idx: int = 0
        self._pos: float = 0.0

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self._trial_idx = int(self.rng.integers(0, self.n_trials))
        self._step_idx = 0
        self._pos = 0.0
        obs = np.concatenate([self.X[self._trial_idx, self._step_idx], np.array([self._pos], dtype=np.float32)])
        info = {"trial_idx": self._trial_idx, "t": self._step_idx, "label": int(self.y[self._trial_idx])}
        return obs.copy(), info

    def step(self, action: float) -> CursorStep:
        if self._trial_idx is None:
            raise RuntimeError("Call reset() before step().")
        a = float(np.clip(action, -1.0, 1.0))
        label = int(self.y[self._trial_idx])
        target = -1.0 if label == 0 else 1.0

        prev_dist = abs(target - self._pos)
        self._pos = float(np.clip(self._pos + self.action_scale * a, -self.max_abs_pos, self.max_abs_pos))
        new_dist = abs(target - self._pos)

        reward = (prev_dist - new_dist) - self.time_cost
        self._step_idx += 1

        success = new_dist <= self.success_tol
        done = success or (self._step_idx >= self.n_steps)
        if success:
            reward += self.success_bonus
        if done:
            obs = np.zeros((self.obs_dim + 1,), dtype=np.float32)
        else:
            obs = np.concatenate([self.X[self._trial_idx, self._step_idx], np.array([self._pos], dtype=np.float32)])

        info = {
            "trial_idx": int(self._trial_idx),
            "t": int(self._step_idx),
            "label": label,
            "target": target,
            "pos": float(self._pos),
            "success": bool(success),
            "done": bool(done),
        }
        return CursorStep(obs=obs.copy(), reward=float(reward), done=bool(done), info=info)
