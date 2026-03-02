from __future__ import annotations

from pathlib import Path


def plot_spike_raster_example(*, out_path: str | Path) -> None:
    """
    Placeholder for spike raster plotting (added in Phase-2).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("TODO: implement spike raster plotting\n", encoding="utf-8")

