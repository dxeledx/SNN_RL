from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_acc_vs_mdt(*, time_costs: list[float], acc: list[float], mdt_steps: list[float], out_path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.array(mdt_steps, dtype=np.float32)
    y = np.array(acc, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=150)
    sc = ax.scatter(x, y, c=np.array(time_costs, dtype=np.float32), cmap="viridis", s=60)
    ax.plot(x, y, color="black", alpha=0.3, linewidth=1)
    ax.set_xlabel("MDT (steps)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto: Accuracy vs Decision Time")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("time_cost")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_kappa_vs_mdt(
    *, time_costs: list[float], kappa: list[float], mdt_steps: list[float], out_path: str | Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.array(mdt_steps, dtype=np.float32)
    y = np.array(kappa, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=150)
    sc = ax.scatter(x, y, c=np.array(time_costs, dtype=np.float32), cmap="viridis", s=60)
    ax.plot(x, y, color="black", alpha=0.3, linewidth=1)
    ax.set_xlabel("MDT (steps)")
    ax.set_ylabel("Cohen's kappa")
    ax.set_title("Pareto: Kappa vs Decision Time")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("time_cost")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
