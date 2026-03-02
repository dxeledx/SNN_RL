from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from bci_snn_rl.data.io import load_subject_windows
from bci_snn_rl.data.variant import variant_iv2b_from_cfg
from bci_snn_rl.eval.evaluate import eval_stop_and_decide_lda_threshold
from bci_snn_rl.utils.config import load_config, make_run_paths
from bci_snn_rl.utils.logging import save_run_meta, save_yaml
from bci_snn_rl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline pareto for stop-and-decide via LDA thresholding")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", action="extend", nargs="+", default=[], help="YAML override(s), repeatable")
    p.add_argument("--subjects", type=str, default=None, help="Comma-separated subjects override, e.g. 1,2,3")
    p.add_argument("--variant", type=str, default=None, help="Override variant name")
    return p.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot_lda_acc_vs_mdt(*, thresholds: list[float], acc: list[float], mdt_steps: list[float], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.array(mdt_steps, dtype=np.float32)
    y = np.array(acc, dtype=np.float32)
    th = np.array(thresholds, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=150)
    sc = ax.scatter(x, y, c=th, cmap="viridis", s=60)
    ax.plot(x, y, color="black", alpha=0.3, linewidth=1)
    ax.set_xlabel("MDT (steps)")
    ax.set_ylabel("Accuracy")
    ax.set_title("LDA Threshold: Accuracy vs Decision Time")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("threshold")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    if args.subjects:
        cfg.setdefault("data", {})["subjects"] = [int(s) for s in args.subjects.split(",") if s.strip()]

    seed = int(cfg["project"].get("seed", 0))
    set_global_seed(seed)

    run_paths = make_run_paths(cfg["project"]["out_dir"])
    save_yaml(run_paths.config_snapshot_path, cfg)
    save_run_meta(run_paths.meta_path, config_path=str(args.config), overrides=list(args.override or []), seed=seed)

    processed_root = cfg["data"].get("processed_root", "data/processed")
    variant = args.variant or variant_iv2b_from_cfg(cfg)
    subjects = cfg["data"].get("subjects", [1])

    # Load once per subject
    subj_data = []
    for s in subjects:
        tr = load_subject_windows(processed_root=processed_root, variant=variant, subject=int(s), split="train")
        ev = load_subject_windows(processed_root=processed_root, variant=variant, subject=int(s), split="eval")
        subj_data.append((int(s), tr.X, tr.y, ev.X, ev.y))

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    rows: list[dict[str, object]] = []

    for th in thresholds:
        summaries = []
        for _subj, Xtr, ytr, Xev, yev in subj_data:
            summaries.append(
                eval_stop_and_decide_lda_threshold(
                    cfg=cfg,
                    X_train=Xtr,
                    y_train=ytr,
                    X_eval=Xev,
                    y_eval=yev,
                    threshold=float(th),
                )
            )

        rows.append(
            {
                "threshold": float(th),
                "acc": float(np.mean([s.acc for s in summaries])),
                "kappa": float(np.mean([s.kappa for s in summaries])),
                "mdt_steps_mean": float(np.mean([s.mdt_steps_mean for s in summaries])),
                "commit_rate": float(np.mean([s.commit_rate for s in summaries])),
                "return_mean": float(np.mean([s.return_mean for s in summaries])),
            }
        )

    out_csv = run_paths.eval_dir / "lda_threshold_pareto.csv"
    _write_csv(out_csv, rows)

    _plot_lda_acc_vs_mdt(
        thresholds=[float(r["threshold"]) for r in rows],
        acc=[float(r["acc"]) for r in rows],
        mdt_steps=[float(r["mdt_steps_mean"]) for r in rows],
        out_path=run_paths.fig_dir / "lda_acc_mdt.png",
    )
    print(f"[lda_pareto] wrote {out_csv}")


if __name__ == "__main__":
    main()

