from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from bci_snn_rl.data.obs_aug import augment_stop_windows
from bci_snn_rl.data.io import SubjectWindows, load_subject_windows, group_by_subject
from bci_snn_rl.data.variant import variant_iv2b_from_cfg
from bci_snn_rl.models.actor_critic import ActorCritic
from bci_snn_rl.rl.ppo import _make_actor_critic_cfg
from bci_snn_rl.utils.config import load_config, make_run_paths
from bci_snn_rl.utils.logging import CSVLogger, save_run_meta, save_yaml
from bci_snn_rl.utils.seed import set_global_seed


@dataclass(frozen=True)
class ImitationCfg:
    threshold: float
    ignore_index: int = -100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supervised pretrain (imitation): learn STOP policy (CONTINUE/COMMIT) using LDA confidence timing"
    )
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
    *, X: np.ndarray, y: np.ndarray, targets: np.ndarray, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return X[tr_idx], y[tr_idx], targets[tr_idx], X[val_idx], y[val_idx], targets[val_idx]


def compute_commit_steps_from_probs(probs: np.ndarray, *, threshold: float) -> np.ndarray:
    """
    Args:
      probs: [N,S,2] LDA posterior probabilities per step.
    Returns:
      commit_steps: [N] int, first step index where max(p)>=threshold, else last step (S-1)
    """
    if probs.ndim != 3:
        raise ValueError(f"Expected probs [N,S,2], got {probs.shape}")
    n_trials, n_steps, n_cls = probs.shape
    if n_cls != 2:
        raise ValueError(f"Expected 2 classes, got {n_cls}")
    conf = probs.max(axis=-1)  # [N,S]
    commit = np.full((n_trials,), n_steps - 1, dtype=np.int64)
    for i in range(n_trials):
        idx = np.where(conf[i] >= float(threshold))[0]
        if idx.size > 0:
            commit[i] = int(idx[0])
    return commit


def make_stop_imitation_targets(
    *, y: np.ndarray, commit_steps: np.ndarray, n_steps: int, ignore_index: int
) -> np.ndarray:
    """
    Build action targets [N,S] for Stop-and-Decide imitation pretrain:
      - t < commit_step: CONTINUE (0)
      - t == commit_step: COMMIT_LEFT (1) if y=0 else COMMIT_RIGHT (2)
      - t > commit_step: ignore_index (not used in loss)
    """
    y = np.asarray(y, dtype=np.int64)
    commit_steps = np.asarray(commit_steps, dtype=np.int64)
    if y.ndim != 1:
        raise ValueError(f"Expected y [N], got {y.shape}")
    if commit_steps.shape != y.shape:
        raise ValueError(f"commit_steps shape mismatch: {commit_steps.shape} vs y {y.shape}")
    if int(n_steps) <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if np.any(commit_steps < 0) or np.any(commit_steps >= int(n_steps)):
        raise ValueError(f"commit_steps out of range [0,{n_steps-1}]")

    N = int(y.shape[0])
    targets = np.full((N, int(n_steps)), int(ignore_index), dtype=np.int64)
    for i in range(N):
        cs = int(commit_steps[i])
        if cs > 0:
            targets[i, :cs] = 0
        targets[i, cs] = int(y[i]) + 1  # 0/1 -> commit-left/right (1/2)
    return targets


def _fit_lda_probs_per_subject(*, items: dict[int, SubjectWindows]) -> dict[int, np.ndarray]:
    """
    Fit a per-subject LDA (window-level) on that subject's train set, and return probs on the same train set.
    Returns:
      subj -> probs [N,S,2]
    """
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for LDA imitation pretrain") from e

    out: dict[int, np.ndarray] = {}
    for subj, sw in items.items():
        X = np.asarray(sw.X, dtype=np.float32)  # [N,S,D]
        y = np.asarray(sw.y, dtype=np.int64)  # [N]
        _N, S, D = X.shape
        Xtr = X.reshape(-1, D)
        ytr = np.repeat(y, S)
        lda = LinearDiscriminantAnalysis()
        lda.fit(Xtr, ytr)
        probs = []
        for t in range(S):
            probs.append(lda.predict_proba(X[:, t, :]))
        out[int(subj)] = np.stack(probs, axis=1).astype(np.float32, copy=False)  # [N,S,2]
    return out


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

    pre = cfg.get("pretrain")
    if not isinstance(pre, dict):
        raise ValueError("Missing pretrain config (expected top-level 'pretrain:' section)")
    epochs = int(pre["epochs"])
    batch_trials = int(pre["batch_trials"])
    lr = float(pre["lr"])
    weight_decay = float(pre["weight_decay"])
    val_frac = float(pre["val_frac"])

    imit_raw = cfg.get("pretrain_imitation", {})
    if not isinstance(imit_raw, dict):
        raise ValueError("pretrain_imitation must be a mapping")
    imit_cfg = ImitationCfg(
        threshold=float(imit_raw.get("threshold", 0.6)),
        ignore_index=int(imit_raw.get("ignore_index", -100)),
    )

    processed_root = cfg["data"].get("processed_root", "data/processed")
    variant = args.variant or variant_iv2b_from_cfg(cfg)
    subjects = cfg["data"].get("subjects", [1])

    subj_items_list: list[SubjectWindows] = [
        load_subject_windows(processed_root=processed_root, variant=variant, subject=int(s), split="train")
        for s in subjects
    ]
    subj_items = group_by_subject(subj_items_list)

    probs_by_subj = _fit_lda_probs_per_subject(items=subj_items)

    task = cfg.get("task", {})
    add_subject_onehot = bool(task.get("add_subject_onehot", False))
    add_time_feature = bool(task.get("add_time_feature", False))

    # Build per-trial targets per subject, then concat across subjects.
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_t: list[np.ndarray] = []
    for subj, sw in subj_items.items():
        X = np.asarray(sw.X, dtype=np.float32)
        y = np.asarray(sw.y, dtype=np.int64)
        probs = probs_by_subj[int(subj)]
        commit_steps = compute_commit_steps_from_probs(probs, threshold=imit_cfg.threshold)
        targets = make_stop_imitation_targets(
            y=y, commit_steps=commit_steps, n_steps=int(X.shape[1]), ignore_index=imit_cfg.ignore_index
        )
        if add_subject_onehot or add_time_feature:
            X = augment_stop_windows(
                X,
                subject=int(subj),
                subjects=[int(s) for s in subjects],
                add_subject_onehot=add_subject_onehot,
                add_time_feature=add_time_feature,
            )
        all_X.append(X)
        all_y.append(y)
        all_t.append(targets)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    targets_all = np.concatenate(all_t, axis=0)

    X_train, y_train, T_train, X_val, y_val, T_val = _split_train_val(
        X=X_all, y=y_all, targets=targets_all, val_frac=val_frac, seed=seed
    )

    device = torch.device(str(cfg["project"]["device"]))
    obs_dim = int(X_train.shape[-1])
    ac = ActorCritic(obs_dim=obs_dim, action_dim=3, cfg=_make_actor_critic_cfg(cfg)).to(device)
    optimizer = torch.optim.Adam(ac.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=int(imit_cfg.ignore_index))

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
            tb = torch.as_tensor(T_train[idx], device=device, dtype=torch.int64)  # [B,S]

            ac.reset_all_states()
            loss = torch.tensor(0.0, device=device)
            for t in range(S):
                logits, _value, _sr = ac(xb[:, t])
                loss = loss + loss_fn(logits, tb[:, t])
            loss = loss / float(S)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu().item()) * int(idx.shape[0])
            train_count += int(idx.shape[0])

        train_loss = train_loss_sum / float(max(1, train_count))

        # Validation: commit-step action accuracy (targets 1/2 only).
        ac.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            correct = 0
            total = 0
            for start in range(0, int(X_val.shape[0]), batch_trials):
                xb = torch.as_tensor(X_val[start : start + batch_trials], device=device, dtype=torch.float32)
                tb = torch.as_tensor(T_val[start : start + batch_trials], device=device, dtype=torch.int64)

                ac.reset_all_states()
                loss = torch.tensor(0.0, device=device)
                logits_all = []
                for t in range(int(xb.shape[1])):
                    logits, _value, _sr = ac(xb[:, t])
                    logits_all.append(logits)
                    loss = loss + loss_fn(logits, tb[:, t])
                loss = loss / float(max(1, int(xb.shape[1])))
                val_loss_sum += float(loss.detach().cpu().item()) * int(xb.shape[0])

                logits_seq = torch.stack(logits_all, dim=1)  # [B,S,3]
                for i in range(int(xb.shape[0])):
                    idxs = torch.where(tb[i] > 0)[0]
                    if idxs.numel() != 1:
                        raise ValueError("Expected exactly one commit-step label per trial")
                    t0 = int(idxs[0].item())
                    pred_action = int(torch.argmax(logits_seq[i, t0]).item())
                    true_action = int(tb[i, t0].item())
                    correct += int(pred_action == true_action)
                    total += 1

            val_loss = val_loss_sum / float(max(1, int(X_val.shape[0])))
            val_acc = float(correct) / float(max(1, total))

        logger.log({"epoch": int(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss), "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            torch.save(ac.state_dict(), str(run_paths.ckpt_dir / "pretrain_best.pt"))

    torch.save(ac.state_dict(), str(run_paths.ckpt_dir / "pretrain_last.pt"))
    logger.close()
    print(f"[pretrain_imitation] done -> {run_paths.out_dir}")


if __name__ == "__main__":
    main()
