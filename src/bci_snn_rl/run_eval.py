from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from bci_snn_rl.data.obs_aug import augment_stop_windows
from bci_snn_rl.data.io import load_subject_epochs, load_subject_windows
from bci_snn_rl.data.variant import variant_iv2b_from_cfg
from bci_snn_rl.eval.evaluate import eval_csp_lda, eval_cursor_control_1d, eval_stop_and_decide
from bci_snn_rl.utils.config import load_config, make_run_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate checkpoint on eval session (feedback) + baselines")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", action="extend", nargs="+", default=[], help="YAML override(s), repeatable")
    p.add_argument("--subjects", type=str, default=None, help="Comma-separated subjects override, e.g. 1,2,3")
    p.add_argument("--variant", type=str, default=None, help="Override variant name")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (default: <out_dir>/checkpoints/best.pt)")
    return p.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    if args.subjects:
        cfg.setdefault("data", {})["subjects"] = [int(s) for s in args.subjects.split(",") if s.strip()]

    run_paths = make_run_paths(cfg["project"]["out_dir"])
    ckpt = args.checkpoint or str(run_paths.ckpt_dir / "best.pt")
    device = torch.device(str(cfg["project"]["device"]))

    processed_root = cfg["data"].get("processed_root", "data/processed")
    variant = args.variant or variant_iv2b_from_cfg(cfg)
    subjects = cfg["data"].get("subjects", [1])

    rows: list[dict[str, object]] = []
    task_name = str(cfg.get("task", {}).get("name", "stop_and_decide"))
    task = cfg.get("task", {})
    add_subject_onehot = bool(task.get("add_subject_onehot", False))
    add_time_feature = bool(task.get("add_time_feature", False))
    for subject in subjects:
        sw = load_subject_windows(processed_root=processed_root, variant=variant, subject=int(subject), split="eval")
        if task_name == "stop_and_decide":
            if add_subject_onehot or add_time_feature:
                sw = type(sw)(
                    subject=sw.subject,
                    X=augment_stop_windows(
                        sw.X,
                        subject=int(sw.subject),
                        subjects=[int(s) for s in subjects],
                        add_subject_onehot=add_subject_onehot,
                        add_time_feature=add_time_feature,
                    ),
                    y=sw.y,
                )
            summary = eval_stop_and_decide(cfg=cfg, X_eval=sw.X, y_eval=sw.y, checkpoint_path=ckpt, device=device)

            tr_ep = load_subject_epochs(
                processed_root=processed_root, variant=variant, subject=int(subject), split="train"
            )
            ev_ep = load_subject_epochs(processed_root=processed_root, variant=variant, subject=int(subject), split="eval")
            csp_acc, csp_kappa = eval_csp_lda(X_train=tr_ep.X, y_train=tr_ep.y, X_eval=ev_ep.X, y_eval=ev_ep.y)

            rows.append(
                {
                    "subject": int(subject),
                    "n_trials": int(summary.n_trials),
                    "acc": float(summary.acc),
                    "kappa": float(summary.kappa),
                    "mdt_steps_mean": float(summary.mdt_steps_mean),
                    "commit_rate": float(summary.commit_rate),
                    "mean_commit_step": float(summary.mean_commit_step),
                    "action_frac_continue": float(summary.action_frac_continue),
                    "action_frac_commit_left": float(summary.action_frac_commit_left),
                    "action_frac_commit_right": float(summary.action_frac_commit_right),
                    "return_mean": float(summary.return_mean),
                    "spike_rate_mean": float(summary.spike_rate_mean),
                    "csp_lda_acc": float(csp_acc),
                    "csp_lda_kappa": float(csp_kappa),
                }
            )
        elif task_name == "cursor_1d":
            summary = eval_cursor_control_1d(cfg=cfg, X_eval=sw.X, y_eval=sw.y, checkpoint_path=ckpt, device=device)
            rows.append(
                {
                    "subject": int(subject),
                    "n_trials": int(summary.n_trials),
                    "return_mean": float(summary.return_mean),
                    "success_rate": float(summary.success_rate),
                    "final_dist_mean": float(summary.final_dist_mean),
                    "steps_mean": float(summary.steps_mean),
                    "spike_rate_mean": float(summary.spike_rate_mean),
                }
            )
        else:
            raise ValueError(f"Unknown task.name: {task_name}")

    per_subj_path = run_paths.eval_dir / "metrics_per_subject.csv"
    _write_csv(per_subj_path, rows)

    # Summary
    def _mean(key: str) -> float:
        return float(np.mean([float(r[key]) for r in rows])) if rows else float("nan")

    def _std(key: str) -> float:
        return float(np.std([float(r[key]) for r in rows])) if rows else float("nan")

    if task_name == "stop_and_decide":
        summary_rows = [
            {
                "metric": "acc",
                "mean": _mean("acc"),
                "std": _std("acc"),
            },
            {"metric": "kappa", "mean": _mean("kappa"), "std": _std("kappa")},
            {"metric": "mdt_steps_mean", "mean": _mean("mdt_steps_mean"), "std": _std("mdt_steps_mean")},
            {"metric": "commit_rate", "mean": _mean("commit_rate"), "std": _std("commit_rate")},
            {"metric": "mean_commit_step", "mean": _mean("mean_commit_step"), "std": _std("mean_commit_step")},
            {
                "metric": "action_frac_continue",
                "mean": _mean("action_frac_continue"),
                "std": _std("action_frac_continue"),
            },
            {
                "metric": "action_frac_commit_left",
                "mean": _mean("action_frac_commit_left"),
                "std": _std("action_frac_commit_left"),
            },
            {
                "metric": "action_frac_commit_right",
                "mean": _mean("action_frac_commit_right"),
                "std": _std("action_frac_commit_right"),
            },
            {"metric": "return_mean", "mean": _mean("return_mean"), "std": _std("return_mean")},
            {"metric": "spike_rate_mean", "mean": _mean("spike_rate_mean"), "std": _std("spike_rate_mean")},
            {"metric": "csp_lda_acc", "mean": _mean("csp_lda_acc"), "std": _std("csp_lda_acc")},
            {"metric": "csp_lda_kappa", "mean": _mean("csp_lda_kappa"), "std": _std("csp_lda_kappa")},
        ]
    elif task_name == "cursor_1d":
        summary_rows = [
            {"metric": "return_mean", "mean": _mean("return_mean"), "std": _std("return_mean")},
            {"metric": "success_rate", "mean": _mean("success_rate"), "std": _std("success_rate")},
            {"metric": "final_dist_mean", "mean": _mean("final_dist_mean"), "std": _std("final_dist_mean")},
            {"metric": "steps_mean", "mean": _mean("steps_mean"), "std": _std("steps_mean")},
            {"metric": "spike_rate_mean", "mean": _mean("spike_rate_mean"), "std": _std("spike_rate_mean")},
        ]
    else:  # pragma: no cover
        raise ValueError(f"Unknown task.name: {task_name}")
    _write_csv(run_paths.eval_dir / "summary.csv", summary_rows)
    print(f"[eval] wrote {per_subj_path}")


if __name__ == "__main__":
    main()
