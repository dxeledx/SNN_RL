from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StopAndDecideEpisode:
    correct: bool
    pred: int | None
    label: int
    steps: int
    total_reward: float
    spike_rate_mean: float


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score

        return float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        # Fallback: report NaN if sklearn isn't installed.
        return float("nan")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())

