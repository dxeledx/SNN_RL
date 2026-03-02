from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParetoPoint:
    time_cost: float
    acc: float
    kappa: float
    mdt_steps_mean: float
    spike_rate_mean: float


def summarize_points(points: list[ParetoPoint]) -> dict[str, Any]:
    return {
        "n_points": len(points),
        "time_cost": [p.time_cost for p in points],
        "acc": [p.acc for p in points],
        "kappa": [p.kappa for p in points],
        "mdt_steps_mean": [p.mdt_steps_mean for p in points],
        "spike_rate_mean": [p.spike_rate_mean for p in points],
    }

