from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bci_snn_rl.data.obs_aug import augment_stop_windows
from bci_snn_rl.data.io import SubjectWindows, concat_windows, load_subject_windows
from bci_snn_rl.data.variant import variant_iv2b_from_cfg
from bci_snn_rl.models.actor_critic import ActorCritic
from bci_snn_rl.rl.ppo import _make_actor_critic_cfg
from bci_snn_rl.utils.config import load_config, make_run_paths
from bci_snn_rl.utils.logging import CSVLogger, save_run_meta, save_yaml
from bci_snn_rl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised pretrain: learn MI decoding (commit-left/right) from train windows")
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
    if any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty out_dir={out_dir} (set project.overwrite=true)")


def _split_train_val(
    *, X: np.ndarray, y: np.ndarray, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < float(val_frac) < 1.0):
        raise ValueError(f"pretrain.val_frac must be in (0,1), got {val_frac}")
    rng = np.random.default_rng(int(seed))
    n = int(X.shape[0])
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * float(val_frac))))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    if tr_idx.size == 0:
        raise ValueError("Not enough trials after train/val split")
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


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
    X_all, y_all = concat_windows(items)
    X_all = np.asarray(X_all, dtype=np.float32)
    y_all = np.asarray(y_all, dtype=np.int64)

    pre = cfg.get("pretrain")
    if not isinstance(pre, dict):
        raise ValueError("Missing pretrain config (expected top-level 'pretrain:' section)")
    epochs = int(pre["epochs"])
    batch_trials = int(pre["batch_trials"])
    lr = float(pre["lr"])
    weight_decay = float(pre["weight_decay"])
    val_frac = float(pre["val_frac"])

    X_train, y_train, X_val, y_val = _split_train_val(X=X_all, y=y_all, val_frac=val_frac, seed=seed)

    device = torch.device(str(cfg["project"]["device"]))
    obs_dim = int(X_train.shape[-1])
    ac = ActorCritic(obs_dim=obs_dim, action_dim=3, cfg=_make_actor_critic_cfg(cfg)).to(device)
    optimizer = torch.optim.Adam(ac.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    logger = CSVLogger(out_dir / "metrics_pretrain.csv")
    best_val_acc = -1.0

    S = int(X_train.shape[1])

    for epoch in range(epochs):
        ac.train()
        perm = np.random.permutation(int(X_train.shape[0]))
        train_loss_sum = 0.0
        train_count = 0

        for start in range(0, int(perm.shape[0]), batch_trials):
            idx = perm[start : start + batch_trials]
            xb = torch.as_tensor(X_train[idx], device=device, dtype=torch.float32)  # [B,S,D]
            yb = torch.as_tensor(y_train[idx], device=device, dtype=torch.int64)  # [B]
            target = yb + 1  # 0/1 -> commit-left/right (1/2)

            ac.reset_all_states()
            loss = torch.tensor(0.0, device=device)
            for t in range(S):
                logits, _value, _sr = ac(xb[:, t])
                loss = loss + loss_fn(logits, target)
            loss = loss / float(S)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu().item()) * int(idx.shape[0])
            train_count += int(idx.shape[0])

        train_loss = train_loss_sum / float(max(1, train_count))

        # Validation
        ac.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            correct = 0
            total = 0
            for start in range(0, int(X_val.shape[0]), batch_trials):
                xb = torch.as_tensor(X_val[start : start + batch_trials], device=device, dtype=torch.float32)
                yb = torch.as_tensor(y_val[start : start + batch_trials], device=device, dtype=torch.int64)
                target = yb + 1

                ac.reset_all_states()
                loss = torch.tensor(0.0, device=device)
                for t in range(int(xb.shape[1])):
                    logits, _value, _sr = ac(xb[:, t])
                    loss = loss + loss_fn(logits, target)
                    pred_action = torch.argmax(logits, dim=-1)  # 0/1/2
                    pred_label = pred_action - 1
                    correct += int(((pred_action != 0) & (pred_label == yb)).sum().item())
                    total += int(pred_action.shape[0])
                loss = loss / float(max(1, int(xb.shape[1])))
                val_loss_sum += float(loss.detach().cpu().item()) * int(xb.shape[0])

            val_loss = val_loss_sum / float(max(1, int(X_val.shape[0])))
            val_acc = float(correct) / float(max(1, total))

        logger.log(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            torch.save(ac.state_dict(), str(run_paths.ckpt_dir / "pretrain_best.pt"))

    torch.save(ac.state_dict(), str(run_paths.ckpt_dir / "pretrain_last.pt"))
    logger.close()
    print(f"[pretrain] done -> {run_paths.out_dir}")


if __name__ == "__main__":
    main()
