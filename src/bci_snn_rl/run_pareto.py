from __future__ import annotations

import argparse
import csv
import copy
from pathlib import Path

import numpy as np
import torch

from bci_snn_rl.data.obs_aug import augment_stop_windows
from bci_snn_rl.data.io import concat_windows, load_subject_windows
from bci_snn_rl.data.variant import variant_iv2b_from_cfg
from bci_snn_rl.eval.evaluate import eval_stop_and_decide
from bci_snn_rl.eval.pareto import ParetoPoint
from bci_snn_rl.rl.ppo import train_stop_and_decide_ppo
from bci_snn_rl.utils.config import deep_update, load_config, make_run_paths
from bci_snn_rl.utils.logging import CSVLogger, save_run_meta, save_yaml
from bci_snn_rl.utils.seed import set_global_seed
from bci_snn_rl.viz.plot_pareto import plot_acc_vs_mdt, plot_kappa_vs_mdt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train+eval multiple time_cost values to produce Pareto curve")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", action="extend", nargs="+", default=[], help="YAML override(s), repeatable")
    p.add_argument("--subjects", type=str, default=None, help="Comma-separated subjects override, e.g. 1,2,3")
    p.add_argument("--variant", type=str, default=None, help="Override variant name")
    return p.parse_args()


def _write_points(path: Path, points: list[ParetoPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["time_cost", "acc", "kappa", "mdt_steps_mean", "spike_rate_mean"],
        )
        w.writeheader()
        for pnt in points:
            w.writerow(
                {
                    "time_cost": pnt.time_cost,
                    "acc": pnt.acc,
                    "kappa": pnt.kappa,
                    "mdt_steps_mean": pnt.mdt_steps_mean,
                    "spike_rate_mean": pnt.spike_rate_mean,
                }
            )


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config, overrides=args.override)
    if args.subjects:
        base_cfg.setdefault("data", {})["subjects"] = [int(s) for s in args.subjects.split(",") if s.strip()]

    seed = int(base_cfg["project"].get("seed", 0))
    set_global_seed(seed)

    base_run = make_run_paths(base_cfg["project"]["out_dir"])
    save_yaml(base_run.config_snapshot_path, base_cfg)
    save_run_meta(base_run.meta_path, config_path=str(args.config), overrides=list(args.override or []), seed=seed)

    processed_root = base_cfg["data"].get("processed_root", "data/processed")
    variant = args.variant or variant_iv2b_from_cfg(base_cfg)
    subjects = base_cfg["data"].get("subjects", [1])

    # Load once
    train_items = [
        load_subject_windows(processed_root=processed_root, variant=variant, subject=int(s), split="train")
        for s in subjects
    ]
    task = base_cfg.get("task", {})
    add_subject_onehot = bool(task.get("add_subject_onehot", False))
    add_time_feature = bool(task.get("add_time_feature", False))
    if add_subject_onehot or add_time_feature:
        train_items = [
            type(it)(
                subject=it.subject,
                X=augment_stop_windows(
                    it.X,
                    subject=int(it.subject),
                    subjects=[int(s) for s in subjects],
                    add_subject_onehot=add_subject_onehot,
                    add_time_feature=add_time_feature,
                ),
                y=it.y,
            )
            for it in train_items
        ]
    X_train, y_train = concat_windows(train_items)

    eval_items = [
        load_subject_windows(processed_root=processed_root, variant=variant, subject=int(s), split="eval")
        for s in subjects
    ]
    if add_subject_onehot or add_time_feature:
        eval_items = [
            type(it)(
                subject=it.subject,
                X=augment_stop_windows(
                    it.X,
                    subject=int(it.subject),
                    subjects=[int(s) for s in subjects],
                    add_subject_onehot=add_subject_onehot,
                    add_time_feature=add_time_feature,
                ),
                y=it.y,
            )
            for it in eval_items
        ]

    device = torch.device(str(base_cfg["project"]["device"]))
    time_costs = list(base_cfg.get("eval", {}).get("pareto", {}).get("time_costs", []))
    if not time_costs:
        raise ValueError("No eval.pareto.time_costs configured")

    points: list[ParetoPoint] = []
    for tc in time_costs:
        tc = float(tc)
        cfg = copy.deepcopy(base_cfg)
        cfg = deep_update(cfg, {"task": {"time_cost": tc}})
        cfg = deep_update(cfg, {"project": {"out_dir": str(base_run.out_dir / "pareto" / f"tc{tc:g}"), "overwrite": True}})

        run = make_run_paths(cfg["project"]["out_dir"])
        save_yaml(run.config_snapshot_path, cfg)

        train_logger = CSVLogger(run.train_metrics_path)
        train_stop_and_decide_ppo(
            cfg=cfg,
            X_train=np.asarray(X_train, dtype=np.float32),
            y_train=np.asarray(y_train, dtype=np.int64),
            device=device,
            logger=train_logger,
            ckpt_dir=run.ckpt_dir,
        )
        train_logger.close()

        # Evaluate on each subject and average
        subj_summaries = []
        for sw in eval_items:
            subj_summaries.append(
                eval_stop_and_decide(
                    cfg=cfg,
                    X_eval=sw.X,
                    y_eval=sw.y,
                    checkpoint_path=str(run.ckpt_dir / "best.pt"),
                    device=device,
                )
            )
        acc = float(np.mean([s.acc for s in subj_summaries]))
        kappa = float(np.mean([s.kappa for s in subj_summaries]))
        mdt = float(np.mean([s.mdt_steps_mean for s in subj_summaries]))
        sr = float(np.mean([s.spike_rate_mean for s in subj_summaries]))
        points.append(ParetoPoint(time_cost=tc, acc=acc, kappa=kappa, mdt_steps_mean=mdt, spike_rate_mean=sr))

    pareto_dir = base_run.out_dir / "pareto"
    _write_points(pareto_dir / "pareto_summary.csv", points)
    plot_acc_vs_mdt(
        time_costs=[p.time_cost for p in points],
        acc=[p.acc for p in points],
        mdt_steps=[p.mdt_steps_mean for p in points],
        out_path=base_run.fig_dir / "pareto_acc_mdt.png",
    )
    plot_kappa_vs_mdt(
        time_costs=[p.time_cost for p in points],
        kappa=[p.kappa for p in points],
        mdt_steps=[p.mdt_steps_mean for p in points],
        out_path=base_run.fig_dir / "pareto_kappa_mdt.png",
    )
    print(f"[pareto] wrote {pareto_dir / 'pareto_summary.csv'}")


if __name__ == "__main__":
    main()
