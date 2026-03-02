from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from bci_snn_rl.data.obs_aug import augment_stop_windows
from bci_snn_rl.data.io import SubjectWindows, concat_windows, load_subject_windows
from bci_snn_rl.data.variant import variant_iv2b_from_cfg
from bci_snn_rl.rl.ppo import train_cursor_control_1d_ppo, train_stop_and_decide_ppo
from bci_snn_rl.utils.config import load_config, make_run_paths
from bci_snn_rl.utils.logging import CSVLogger, save_run_meta, save_yaml
from bci_snn_rl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO agent (SNN actor by default) on stop-and-decide env")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", action="extend", nargs="+", default=[], help="YAML override(s), repeatable")
    p.add_argument("--subjects", type=str, default=None, help="Comma-separated subjects override, e.g. 1,2,3")
    p.add_argument("--variant", type=str, default=None, help="Override variant name")
    return p.parse_args()


def _assert_can_write(out_dir: Path, *, overwrite: bool) -> None:
    if not out_dir.exists():
        return
    if overwrite:
        return
    # "exists" but allow empty dir
    if any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty out_dir={out_dir} (set project.overwrite=true)")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    if args.subjects:
        cfg.setdefault("data", {})["subjects"] = [int(s) for s in args.subjects.split(",") if s.strip()]

    seed = int(cfg["project"].get("seed", 0))
    set_global_seed(seed)

    out_dir = Path(cfg["project"]["out_dir"])
    _assert_can_write(out_dir, overwrite=bool(cfg["project"].get("overwrite", False)))
    run_paths = make_run_paths(out_dir)

    save_yaml(run_paths.config_snapshot_path, cfg)
    save_run_meta(run_paths.meta_path, config_path=str(args.config), overrides=list(args.override or []), seed=seed)

    processed_root = cfg["data"].get("processed_root", "data/processed")
    variant = args.variant or variant_iv2b_from_cfg(cfg)
    subjects = cfg["data"].get("subjects", [1])

    items: list[SubjectWindows] = [
        load_subject_windows(processed_root=processed_root, variant=variant, subject=int(s), split="train")
        for s in subjects
    ]

    task_name = str(cfg.get("task", {}).get("name", "stop_and_decide"))
    if task_name == "stop_and_decide":
        task = cfg.get("task", {})
        add_subject_onehot = bool(task.get("add_subject_onehot", False))
        add_time_feature = bool(task.get("add_time_feature", False))
        if add_subject_onehot or add_time_feature:
            items = [
                SubjectWindows(
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
                for it in items
            ]
    X_train, y_train = concat_windows(items)

    device = torch.device(str(cfg["project"]["device"]))
    logger = CSVLogger(run_paths.train_metrics_path)
    if task_name == "stop_and_decide":
        train_stop_and_decide_ppo(
            cfg=cfg,
            X_train=np.asarray(X_train, dtype=np.float32),
            y_train=np.asarray(y_train, dtype=np.int64),
            device=device,
            logger=logger,
            ckpt_dir=run_paths.ckpt_dir,
        )
    elif task_name == "cursor_1d":
        train_cursor_control_1d_ppo(
            cfg=cfg,
            X_train=np.asarray(X_train, dtype=np.float32),
            y_train=np.asarray(y_train, dtype=np.int64),
            device=device,
            logger=logger,
            ckpt_dir=run_paths.ckpt_dir,
        )
    else:
        raise ValueError(f"Unknown task.name: {task_name}")
    logger.close()
    print(f"[train] done -> {run_paths.out_dir}")


if __name__ == "__main__":
    main()
