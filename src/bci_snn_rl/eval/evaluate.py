from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from bci_snn_rl.envs.stop_and_decide import ACTION_COMMIT_LEFT, ACTION_COMMIT_RIGHT, ACTION_CONTINUE
from bci_snn_rl.eval.metrics import StopAndDecideEpisode, accuracy, cohen_kappa
from bci_snn_rl.models.actor_critic import ActorCritic, ActorCriticConfig, GaussianActorCritic
from bci_snn_rl.rl.ppo import _make_actor_critic_cfg
from bci_snn_rl.utils.stop_policy import mask_continue_at_last_step


@dataclass(frozen=True)
class StopAndDecideSummary:
    n_trials: int
    acc: float
    kappa: float
    mdt_steps_mean: float
    commit_rate: float
    mean_commit_step: float
    action_frac_continue: float
    action_frac_commit_left: float
    action_frac_commit_right: float
    return_mean: float
    spike_rate_mean: float


@dataclass(frozen=True)
class CursorControl1DSummary:
    n_trials: int
    return_mean: float
    success_rate: float
    final_dist_mean: float
    steps_mean: float
    spike_rate_mean: float


def eval_stop_and_decide(
    *,
    cfg: dict[str, Any],
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    checkpoint_path: str,
    device: torch.device,
) -> StopAndDecideSummary:
    task = cfg["task"]
    time_cost = float(task["time_cost"])
    no_commit_penalty = float(task.get("no_commit_penalty", -1.0))
    force_commit_last_step = bool(task.get("force_commit_last_step", False))
    if force_commit_last_step and not bool(task.get("add_time_feature", False)):
        raise ValueError("task.force_commit_last_step requires task.add_time_feature=true (t_norm at obs[-1])")

    obs_dim = int(X_eval.shape[-1])
    ac = ActorCritic(obs_dim=obs_dim, action_dim=3, cfg=_make_actor_critic_cfg(cfg)).to(device)
    ac.load_state_dict(torch.load(checkpoint_path, map_location=device))
    ac.eval()

    episodes: list[StopAndDecideEpisode] = []
    preds: list[int] = []
    labels: list[int] = []
    action_counts = np.zeros((3,), dtype=np.int64)
    commit_steps: list[int] = []

    for i in range(int(X_eval.shape[0])):
        label = int(y_eval[i])
        steps = 0
        total_reward = 0.0
        spike_rates: list[float] = []

        # Reset spiking state for single-item batch.
        ac.reset_done(torch.tensor([True], device=device))

        pred: int | None = None
        for t in range(int(X_eval.shape[1])):
            obs_t = torch.as_tensor(X_eval[i, t][None, :], device=device, dtype=torch.float32)
            with torch.no_grad():
                logits, _value, spike_rate = ac(obs_t)
                if force_commit_last_step:
                    logits = mask_continue_at_last_step(logits, obs_t)
                spike_rates.append(float(spike_rate.detach().cpu().item()))
                action = int(torch.argmax(logits, dim=-1).item())
            action_counts[action] += 1

            steps += 1
            total_reward += -time_cost

            if action == ACTION_CONTINUE:
                if t >= int(X_eval.shape[1] - 1):
                    total_reward += no_commit_penalty
                    pred = None
                    break
                continue

            if action == ACTION_COMMIT_LEFT:
                pred = 0
            elif action == ACTION_COMMIT_RIGHT:
                pred = 1
            else:
                raise ValueError(f"Invalid action from policy: {action}")

            commit_steps.append(int(steps))
            total_reward += 1.0 if pred == label else -1.0
            break

        correct = (pred == label)
        episodes.append(
            StopAndDecideEpisode(
                correct=bool(correct),
                pred=pred,
                label=label,
                steps=steps,
                total_reward=float(total_reward),
                spike_rate_mean=float(np.mean(spike_rates)) if spike_rates else 0.0,
            )
        )
        if pred is not None:
            preds.append(int(pred))
            labels.append(int(label))

    # If some episodes never commit, treat as wrong class 0 for accuracy/kappa calculation.
    y_true = np.array([ep.label for ep in episodes], dtype=np.int64)
    y_pred = np.array([(ep.pred if ep.pred is not None else 0) for ep in episodes], dtype=np.int64)

    acc = accuracy(y_true, y_pred)
    kappa = cohen_kappa(y_true, y_pred)
    mdt_steps_mean = float(np.mean([ep.steps for ep in episodes])) if episodes else float("nan")
    commit_rate = float(len(commit_steps) / float(len(episodes))) if episodes else 0.0
    mean_commit_step = float(np.mean(commit_steps)) if commit_steps else float("nan")
    total_actions = float(action_counts.sum())
    if total_actions > 0:
        action_fracs = action_counts.astype(np.float64) / total_actions
    else:
        action_fracs = np.zeros_like(action_counts, dtype=np.float64)
    return_mean = float(np.mean([ep.total_reward for ep in episodes])) if episodes else float("nan")
    spike_rate_mean = float(np.mean([ep.spike_rate_mean for ep in episodes])) if episodes else float("nan")

    return StopAndDecideSummary(
        n_trials=int(len(episodes)),
        acc=float(acc),
        kappa=float(kappa),
        mdt_steps_mean=float(mdt_steps_mean),
        commit_rate=float(commit_rate),
        mean_commit_step=float(mean_commit_step),
        action_frac_continue=float(action_fracs[ACTION_CONTINUE]),
        action_frac_commit_left=float(action_fracs[ACTION_COMMIT_LEFT]),
        action_frac_commit_right=float(action_fracs[ACTION_COMMIT_RIGHT]),
        return_mean=float(return_mean),
        spike_rate_mean=float(spike_rate_mean),
    )


def eval_stop_and_decide_lda_threshold(
    *,
    cfg: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    threshold: float,
) -> StopAndDecideSummary:
    """
    Classical "closed-loop" baseline using LDA window-level posteriors:
      - Fit LDA on train windows (flattened N*S).
      - On eval, at each step t compute p(y|x_t).
      - If max(p) >= threshold, commit to argmax class.
      - If never reaches threshold, force commit at the last step.
    """
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for LDA baseline") from e

    task = cfg["task"]
    time_cost = float(task["time_cost"])

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_eval = np.asarray(X_eval, dtype=np.float32)
    y_eval = np.asarray(y_eval, dtype=np.int64)

    n_steps = int(X_train.shape[1])
    obs_dim = int(X_train.shape[-1])

    Xtr = X_train.reshape(-1, obs_dim)
    ytr = np.repeat(y_train, n_steps)
    lda = LinearDiscriminantAnalysis()
    lda.fit(Xtr, ytr)

    episodes: list[StopAndDecideEpisode] = []
    action_counts = np.zeros((3,), dtype=np.int64)
    commit_steps: list[int] = []

    for i in range(int(X_eval.shape[0])):
        label = int(y_eval[i])
        steps = 0
        total_reward = 0.0
        pred: int | None = None

        for t in range(int(X_eval.shape[1])):
            probs = lda.predict_proba(X_eval[i, t][None, :])[0]
            pred_cls = int(np.argmax(probs))
            conf = float(np.max(probs))

            steps += 1
            total_reward += -time_cost

            force_commit = (t == int(X_eval.shape[1] - 1))
            if (conf >= float(threshold)) or force_commit:
                pred = int(pred_cls)
                action = ACTION_COMMIT_LEFT if pred == 0 else ACTION_COMMIT_RIGHT
                action_counts[action] += 1
                commit_steps.append(int(steps))
                total_reward += 1.0 if pred == label else -1.0
                break

            action_counts[ACTION_CONTINUE] += 1

        correct = (pred == label)
        episodes.append(
            StopAndDecideEpisode(
                correct=bool(correct),
                pred=pred,
                label=label,
                steps=steps,
                total_reward=float(total_reward),
                spike_rate_mean=0.0,
            )
        )

    y_true = np.array([ep.label for ep in episodes], dtype=np.int64)
    y_pred = np.array([(ep.pred if ep.pred is not None else 0) for ep in episodes], dtype=np.int64)

    acc = accuracy(y_true, y_pred)
    kappa = cohen_kappa(y_true, y_pred)
    mdt_steps_mean = float(np.mean([ep.steps for ep in episodes])) if episodes else float("nan")
    commit_rate = float(len(commit_steps) / float(len(episodes))) if episodes else 0.0
    mean_commit_step = float(np.mean(commit_steps)) if commit_steps else float("nan")
    total_actions = float(action_counts.sum())
    if total_actions > 0:
        action_fracs = action_counts.astype(np.float64) / total_actions
    else:
        action_fracs = np.zeros_like(action_counts, dtype=np.float64)
    return_mean = float(np.mean([ep.total_reward for ep in episodes])) if episodes else float("nan")

    return StopAndDecideSummary(
        n_trials=int(len(episodes)),
        acc=float(acc),
        kappa=float(kappa),
        mdt_steps_mean=float(mdt_steps_mean),
        commit_rate=float(commit_rate),
        mean_commit_step=float(mean_commit_step),
        action_frac_continue=float(action_fracs[ACTION_CONTINUE]),
        action_frac_commit_left=float(action_fracs[ACTION_COMMIT_LEFT]),
        action_frac_commit_right=float(action_fracs[ACTION_COMMIT_RIGHT]),
        return_mean=float(return_mean),
        spike_rate_mean=0.0,
    )


def eval_cursor_control_1d(
    *,
    cfg: dict[str, Any],
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    checkpoint_path: str,
    device: torch.device,
) -> CursorControl1DSummary:
    task = cfg["task"]
    time_cost = float(task["time_cost"])
    success_tol = float(task.get("success_tol", 0.1))
    max_abs_pos = float(task.get("max_abs_pos", 2.0))
    action_scale = float(task.get("action_scale", 0.2))
    success_bonus = float(task.get("success_bonus", 1.0))

    obs_dim = int(X_eval.shape[-1]) + 1
    ac = GaussianActorCritic(obs_dim=obs_dim, action_dim=1, cfg=_make_actor_critic_cfg(cfg)).to(device)
    ac.load_state_dict(torch.load(checkpoint_path, map_location=device))
    ac.eval()

    returns: list[float] = []
    successes: list[bool] = []
    final_dists: list[float] = []
    steps_list: list[int] = []
    spike_rates_all: list[float] = []

    for i in range(int(X_eval.shape[0])):
        label = int(y_eval[i])
        target = -1.0 if label == 0 else 1.0
        pos = 0.0
        total_reward = 0.0
        spike_rates: list[float] = []

        ac.reset_done(torch.tensor([True], device=device))

        success = False
        final_dist = abs(target - pos)
        steps = 0

        for t in range(int(X_eval.shape[1])):
            obs_np = np.concatenate([X_eval[i, t], np.array([pos], dtype=np.float32)], axis=0)[None, :]
            obs_t = torch.as_tensor(obs_np, device=device, dtype=torch.float32)
            with torch.no_grad():
                mean, _log_std, _value, spike_rate = ac(obs_t)
                spike_rates.append(float(spike_rate.detach().cpu().item()))
                action = float(torch.tanh(mean)[0, 0].detach().cpu().item())

            prev_dist = abs(target - pos)
            pos = float(np.clip(pos + action_scale * action, -max_abs_pos, max_abs_pos))
            final_dist = abs(target - pos)

            total_reward += (prev_dist - final_dist) - time_cost
            steps += 1

            if final_dist <= success_tol:
                success = True
                total_reward += success_bonus
                break

        returns.append(float(total_reward))
        successes.append(bool(success))
        final_dists.append(float(final_dist))
        steps_list.append(int(steps))
        spike_rates_all.append(float(np.mean(spike_rates)) if spike_rates else 0.0)

    return CursorControl1DSummary(
        n_trials=int(X_eval.shape[0]),
        return_mean=float(np.mean(returns)) if returns else float("nan"),
        success_rate=float(np.mean(successes)) if successes else float("nan"),
        final_dist_mean=float(np.mean(final_dists)) if final_dists else float("nan"),
        steps_mean=float(np.mean(steps_list)) if steps_list else float("nan"),
        spike_rate_mean=float(np.mean(spike_rates_all)) if spike_rates_all else float("nan"),
    )


def eval_csp_lda(
    *, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray
) -> tuple[float, float]:
    """
    Classical baseline: CSP + LDA (trial-level).
    Returns: (acc, kappa)
    """
    try:
        import mne
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependencies for CSP/LDA baseline: mne, scikit-learn") from e

    # MNE CSP expects shape [n_epochs, n_channels, n_times]
    #
    # NOTE: recent MNE versions may require float64 input for some covariance/rank
    # computations (they internally call RawArray(copy=None) which forbids casting).
    # To keep this baseline robust, cast explicitly here.
    X_train = np.asarray(X_train, dtype=np.float64)
    X_eval = np.asarray(X_eval, dtype=np.float64)

    csp = CSP(n_components=min(6, X_train.shape[1]), reg=None, log=True, norm_trace=False)
    clf = LinearDiscriminantAnalysis()

    X_train_csp = csp.fit_transform(X_train, y_train)
    clf.fit(X_train_csp, y_train)
    X_eval_csp = csp.transform(X_eval)
    y_pred = clf.predict(X_eval_csp)

    acc = float((y_pred == y_eval).mean())
    kappa = cohen_kappa(y_eval.astype(np.int64), y_pred.astype(np.int64))
    return acc, kappa
