from __future__ import annotations

from pathlib import Path


def plot_cursor_traj_example(*, out_path: str | Path) -> None:
    """
    Placeholder for cursor-control trajectory plots (added in Phase-2).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("TODO: implement cursor trajectory plotting\n", encoding="utf-8")

