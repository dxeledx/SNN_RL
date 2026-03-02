from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VecStep:
    obs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    infos: list[dict[str, Any]]


class SimpleVectorEnv:
    """
    A minimal vector wrapper that supports per-env autoreset.

    (We avoid Gymnasium's vector env so we can precisely control spiking-state resets.)
    """

    def __init__(self, env_fns: list[Callable[[], Any]]) -> None:
        if not env_fns:
            raise ValueError("env_fns must be non-empty")
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def reset(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        obs_list = []
        infos: list[dict[str, Any]] = []
        for env in self.envs:
            o, info = env.reset()
            obs_list.append(o)
            infos.append(dict(info))
        return np.stack(obs_list, axis=0).astype(np.float32, copy=False), infos

    def step(self, actions: np.ndarray) -> VecStep:
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected actions shape [{self.num_envs}], got {actions.shape}")
        obs_list = []
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=bool)
        infos: list[dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            step = env.step(actions[i])
            obs_list.append(step.obs)
            rewards[i] = float(step.reward)
            dones[i] = bool(step.done)
            info = dict(step.info)
            if step.done:
                o_reset, info_reset = env.reset()
                obs_list[-1] = o_reset
                info["reset_info"] = dict(info_reset)
            infos.append(info)
        return VecStep(
            obs=np.stack(obs_list, axis=0).astype(np.float32, copy=False),
            rewards=rewards,
            dones=dones,
            infos=infos,
        )

