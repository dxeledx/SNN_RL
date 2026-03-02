from __future__ import annotations

import numpy as np

from bci_snn_rl.envs.stop_and_decide import (
    ACTION_COMMIT_LEFT,
    ACTION_COMMIT_RIGHT,
    ACTION_CONTINUE,
    StopAndDecideEnv,
)


def test_stop_and_decide_commit_reward() -> None:
    # 2 trials, 3 steps, 4-dim obs
    X = np.zeros((2, 3, 4), dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    env = StopAndDecideEnv(X=X, y=y, time_cost=0.1, no_commit_penalty=-1.0, rng=np.random.default_rng(0))
    obs, info = env.reset()
    assert obs.shape == (4,)
    label = info["label"]

    action = ACTION_COMMIT_LEFT if label == 0 else ACTION_COMMIT_RIGHT
    step = env.step(action)
    assert step.done is True
    # reward = -time_cost + (+1)
    assert abs(step.reward - (1.0 - 0.1)) < 1e-6


def test_stop_and_decide_no_commit_penalty() -> None:
    X = np.zeros((1, 2, 4), dtype=np.float32)
    y = np.array([0], dtype=np.int64)
    env = StopAndDecideEnv(X=X, y=y, time_cost=0.0, no_commit_penalty=-2.0, rng=np.random.default_rng(0))
    env.reset()
    s1 = env.step(ACTION_CONTINUE)
    assert s1.done is False
    s2 = env.step(ACTION_CONTINUE)
    assert s2.done is True
    assert abs(s2.reward - (-2.0)) < 1e-6

