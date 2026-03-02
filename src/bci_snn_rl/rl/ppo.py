from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from pathlib import Path

from bci_snn_rl.envs.cursor_control_1d import CursorControl1DEnv
from bci_snn_rl.envs.stop_and_decide import StopAndDecideEnv
from bci_snn_rl.envs.wrappers import SimpleVectorEnv
from bci_snn_rl.models.actor_critic import ActorCritic, ActorCriticConfig, GaussianActorCritic
from bci_snn_rl.rl.buffers import RolloutBuffer, RolloutBufferContinuous
from bci_snn_rl.utils.logging import CSVLogger
from bci_snn_rl.utils.stop_policy import mask_continue_at_last_step


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int
    n_envs: int
    rollout_steps: int
    gamma: float
    gae_lambda: float
    lr: float
    update_epochs: int
    minibatch_size: int
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    spike_rate_coef: float


def spike_homeostasis_loss(spike_mean: torch.Tensor, *, target_rate: float, coef: float) -> torch.Tensor:
    coef_f = float(coef)
    if coef_f == 0.0:
        return spike_mean.new_tensor(0.0)
    return coef_f * (spike_mean - float(target_rate)).pow(2)


def _make_actor_critic_cfg(cfg: dict[str, Any]) -> ActorCriticConfig:
    m = cfg["model"]
    enc = m["encoder"]
    actor = m["actor"]
    critic = m["critic"]
    snn = m.get("snn", {})
    task = cfg.get("task", {})
    subjects = cfg.get("data", {}).get("subjects", [])
    subj_dim = int(len(subjects)) if bool(task.get("add_subject_onehot", False)) else 0
    aux_dim = subj_dim + (1 if bool(task.get("add_time_feature", False)) else 0)
    return ActorCriticConfig(
        encoder_type=str(enc["type"]),
        encoder_learnable_threshold=bool(enc.get("learnable_threshold", True)),
        encoder_theta_init=float(enc.get("theta_init", 0.25)),
        encoder_apply_to=str(enc.get("apply_to", "all")),
        encoder_aux_dim=int(aux_dim),
        snn_input_scale=float(snn.get("input_scale", 1.0)),
        snn_v_threshold=float(snn.get("v_threshold", 1.0)),
        snn_output_mode=str(snn.get("output_mode", "spike")),
        actor_type=str(actor.get("type", "snn_mlp")),
        actor_hidden_sizes=[int(x) for x in actor["hidden_sizes"]],
        critic_hidden_sizes=[int(x) for x in critic["hidden_sizes"]],
    )


def train_stop_and_decide_ppo(
    *,
    cfg: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    logger: CSVLogger,
    ckpt_dir: Path,
) -> dict[str, Any]:
    task = cfg["task"]
    time_cost = float(task["time_cost"])
    no_commit_penalty = float(task.get("no_commit_penalty", -1.0))
    force_commit_last_step = bool(task.get("force_commit_last_step", False))
    if force_commit_last_step and not bool(task.get("add_time_feature", False)):
        raise ValueError("task.force_commit_last_step requires task.add_time_feature=true (t_norm at obs[-1])")

    train_cfg = cfg["train"]
    homeo_cfg = train_cfg.get("spike_homeostasis") or {}
    if not isinstance(homeo_cfg, dict):
        raise TypeError(f"train.spike_homeostasis must be a dict, got {type(homeo_cfg).__name__}")
    homeo_target = float(homeo_cfg.get("target_rate", 0.05))
    homeo_coef = float(homeo_cfg.get("coef", 0.0))
    ppo_cfg = PPOConfig(
        total_steps=int(train_cfg["total_steps"]),
        n_envs=int(train_cfg["n_envs"]),
        rollout_steps=int(train_cfg["rollout_steps"]),
        gamma=float(train_cfg["gamma"]),
        gae_lambda=float(train_cfg["gae_lambda"]),
        lr=float(train_cfg["lr"]),
        update_epochs=int(train_cfg["update_epochs"]),
        minibatch_size=int(train_cfg["minibatch_size"]),
        clip_coef=float(train_cfg["clip_coef"]),
        ent_coef=float(train_cfg["ent_coef"]),
        vf_coef=float(train_cfg["vf_coef"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        spike_rate_coef=float(train_cfg.get("spike_rate_coef", 0.0)),
    )

    obs_dim = int(X_train.shape[-1])
    action_dim = 3

    rng = np.random.default_rng(int(cfg["project"]["seed"]))

    def _env_fn(i: int) -> StopAndDecideEnv:
        return StopAndDecideEnv(
            X=X_train,
            y=y_train,
            time_cost=time_cost,
            no_commit_penalty=no_commit_penalty,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + i),
        )

    venv = SimpleVectorEnv([lambda i=i: _env_fn(i) for i in range(ppo_cfg.n_envs)])

    ac = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, cfg=_make_actor_critic_cfg(cfg)).to(device)
    init_ckpt = train_cfg.get("init_checkpoint")
    if init_ckpt:
        init_path = Path(str(init_ckpt)).expanduser()
        if not init_path.is_file():
            raise FileNotFoundError(f"train.init_checkpoint not found: {init_path}")
        state = torch.load(str(init_path), map_location=device)
        ac.load_state_dict(state, strict=False)
    optimizer = torch.optim.Adam(ac.parameters(), lr=ppo_cfg.lr)

    # Episode stats (for logging / checkpointing)
    ep_returns = np.zeros((ppo_cfg.n_envs,), dtype=np.float32)
    ep_lens = np.zeros((ppo_cfg.n_envs,), dtype=np.int32)
    completed_returns: list[float] = []
    completed_lens: list[int] = []

    best_score = -1e9
    global_step = 0
    num_updates = int(ppo_cfg.total_steps // (ppo_cfg.rollout_steps * ppo_cfg.n_envs))

    for update in range(num_updates):
        # Start each rollout from fresh episodes so policy state is well-defined.
        obs_np, _infos = venv.reset()
        obs = torch.as_tensor(obs_np, device=device, dtype=torch.float32)
        ac.reset_all_states()
        ep_returns[:] = 0.0
        ep_lens[:] = 0

        buf = RolloutBuffer.allocate(T=ppo_cfg.rollout_steps, N=ppo_cfg.n_envs, obs_dim=obs_dim, device=device)

        for _step in range(ppo_cfg.rollout_steps):
            global_step += ppo_cfg.n_envs
            with torch.no_grad():
                logits, values, spike_rate = ac(obs)
                if force_commit_last_step:
                    logits = mask_continue_at_last_step(logits, obs)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            vec_step = venv.step(actions.detach().cpu().numpy())
            next_obs = torch.as_tensor(vec_step.obs, device=device, dtype=torch.float32)
            rewards = torch.as_tensor(vec_step.rewards, device=device, dtype=torch.float32)
            dones = torch.as_tensor(vec_step.dones, device=device, dtype=torch.bool)

            # Update episode statistics
            ep_returns += vec_step.rewards
            ep_lens += 1
            for i, d in enumerate(vec_step.dones.tolist()):
                if d:
                    completed_returns.append(float(ep_returns[i]))
                    completed_lens.append(int(ep_lens[i]))
                    ep_returns[i] = 0.0
                    ep_lens[i] = 0

            buf.add(
                obs=obs,
                actions=actions,
                logprobs=logprobs,
                rewards=rewards,
                dones=dones,
                values=values,
            )

            # Reset spiking state for finished envs (vectorized batch indices).
            ac.reset_done(dones)
            obs = next_obs

        with torch.no_grad():
            _logits, last_values, _sr = ac(obs)

        buf.compute_returns_advantages(last_values=last_values, gamma=ppo_cfg.gamma, gae_lambda=ppo_cfg.gae_lambda)
        advantages = buf.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        clip_fracs = []
        spike_mean_val = 0.0
        homeo_loss_val = 0.0
        for epoch in range(ppo_cfg.update_epochs):
            # Recurrent-style update: replay the rollout sequence in order to match spiking state.
            ac.reset_all_states()

            pg_loss_sum = torch.tensor(0.0, device=device)
            value_loss_sum = torch.tensor(0.0, device=device)
            entropy_sum = torch.tensor(0.0, device=device)
            spike_sum = torch.tensor(0.0, device=device)

            for t in range(ppo_cfg.rollout_steps):
                logits, new_values, spike_rate = ac(buf.obs[t])
                if force_commit_last_step:
                    logits = mask_continue_at_last_step(logits, buf.obs[t])
                dist = torch.distributions.Categorical(logits=logits)
                new_logprob = dist.log_prob(buf.actions[t])
                entropy = dist.entropy().mean()

                ratio = (new_logprob - buf.logprobs[t]).exp()
                adv_t = advantages[t]

                pg_loss1 = -adv_t * ratio
                pg_loss2 = -adv_t * torch.clamp(ratio, 1.0 - ppo_cfg.clip_coef, 1.0 + ppo_cfg.clip_coef)
                pg_loss_t = torch.max(pg_loss1, pg_loss2).mean()

                value_loss_t = 0.5 * (buf.returns[t] - new_values).pow(2).mean()

                pg_loss_sum = pg_loss_sum + pg_loss_t
                value_loss_sum = value_loss_sum + value_loss_t
                entropy_sum = entropy_sum + entropy
                spike_sum = spike_sum + spike_rate
                clip_fracs.append(((ratio - 1.0).abs() > ppo_cfg.clip_coef).float().mean().item())

                # Reset policy state for envs that terminated at this time step.
                ac.reset_done(buf.dones[t])

            pg_loss = pg_loss_sum / float(ppo_cfg.rollout_steps)
            value_loss = value_loss_sum / float(ppo_cfg.rollout_steps)
            entropy_mean = entropy_sum / float(ppo_cfg.rollout_steps)
            spike_mean = spike_sum / float(ppo_cfg.rollout_steps)
            homeo_loss = spike_homeostasis_loss(spike_mean, target_rate=homeo_target, coef=homeo_coef)

            loss = (
                pg_loss
                + ppo_cfg.vf_coef * value_loss
                - ppo_cfg.ent_coef * entropy_mean
                + ppo_cfg.spike_rate_coef * spike_mean
                + homeo_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), ppo_cfg.max_grad_norm)
            optimizer.step()
            spike_mean_val = float(spike_mean.detach().cpu().item())
            homeo_loss_val = float(homeo_loss.detach().cpu().item())

        # Logging + checkpoint
        ep_ret_mean = float(np.mean(completed_returns[-100:])) if completed_returns else float("nan")
        ep_len_mean = float(np.mean(completed_lens[-100:])) if completed_lens else float("nan")
        logger.log(
            {
                "update": update,
                "global_step": global_step,
                "ep_return_mean_100": ep_ret_mean,
                "ep_len_mean_100": ep_len_mean,
                "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
                "spike_mean": float(spike_mean_val),
                "homeo_loss": float(homeo_loss_val),
            }
        )

        score = ep_ret_mean if np.isfinite(ep_ret_mean) else -1e9
        if score > best_score:
            best_score = score
            torch.save(ac.state_dict(), str(ckpt_dir / "best.pt"))

    torch.save(ac.state_dict(), str(ckpt_dir / "last.pt"))
    return {"best_score": float(best_score), "global_step": int(global_step)}


def _tanh_squash_action(dist: torch.distributions.Normal, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Squashed Gaussian action with log-prob correction.
    Returns: (a, logprob) where:
      a in (-1, 1), logprob is summed over action dims -> [N]
    """
    a = torch.tanh(u)
    logprob_u = dist.log_prob(u)
    # Change-of-variables term for tanh.
    log_det = torch.log(1.0 - a.pow(2) + 1e-6)
    logprob = (logprob_u - log_det).sum(dim=-1)
    return a, logprob


def _atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.atanh(x)


def train_cursor_control_1d_ppo(
    *,
    cfg: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    logger: CSVLogger,
    ckpt_dir: Path,
) -> dict[str, Any]:
    task = cfg["task"]
    time_cost = float(task["time_cost"])
    success_tol = float(task.get("success_tol", 0.1))
    max_abs_pos = float(task.get("max_abs_pos", 2.0))
    action_scale = float(task.get("action_scale", 0.2))
    success_bonus = float(task.get("success_bonus", 1.0))

    train_cfg = cfg["train"]
    homeo_cfg = train_cfg.get("spike_homeostasis") or {}
    if not isinstance(homeo_cfg, dict):
        raise TypeError(f"train.spike_homeostasis must be a dict, got {type(homeo_cfg).__name__}")
    homeo_target = float(homeo_cfg.get("target_rate", 0.05))
    homeo_coef = float(homeo_cfg.get("coef", 0.0))
    ppo_cfg = PPOConfig(
        total_steps=int(train_cfg["total_steps"]),
        n_envs=int(train_cfg["n_envs"]),
        rollout_steps=int(train_cfg["rollout_steps"]),
        gamma=float(train_cfg["gamma"]),
        gae_lambda=float(train_cfg["gae_lambda"]),
        lr=float(train_cfg["lr"]),
        update_epochs=int(train_cfg["update_epochs"]),
        minibatch_size=int(train_cfg["minibatch_size"]),
        clip_coef=float(train_cfg["clip_coef"]),
        ent_coef=float(train_cfg["ent_coef"]),
        vf_coef=float(train_cfg["vf_coef"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        spike_rate_coef=float(train_cfg.get("spike_rate_coef", 0.0)),
    )

    obs_dim = int(X_train.shape[-1]) + 1
    action_dim = 1

    rng = np.random.default_rng(int(cfg["project"]["seed"]))

    def _env_fn(i: int) -> CursorControl1DEnv:
        return CursorControl1DEnv(
            X=X_train,
            y=y_train,
            time_cost=time_cost,
            success_tol=success_tol,
            max_abs_pos=max_abs_pos,
            action_scale=action_scale,
            success_bonus=success_bonus,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + i),
        )

    venv = SimpleVectorEnv([lambda i=i: _env_fn(i) for i in range(ppo_cfg.n_envs)])

    ac = GaussianActorCritic(obs_dim=obs_dim, action_dim=action_dim, cfg=_make_actor_critic_cfg(cfg)).to(device)
    init_ckpt = train_cfg.get("init_checkpoint")
    if init_ckpt:
        init_path = Path(str(init_ckpt)).expanduser()
        if not init_path.is_file():
            raise FileNotFoundError(f"train.init_checkpoint not found: {init_path}")
        state = torch.load(str(init_path), map_location=device)
        ac.load_state_dict(state, strict=False)
    optimizer = torch.optim.Adam(ac.parameters(), lr=ppo_cfg.lr)

    ep_returns = np.zeros((ppo_cfg.n_envs,), dtype=np.float32)
    ep_lens = np.zeros((ppo_cfg.n_envs,), dtype=np.int32)
    completed_returns: list[float] = []
    completed_lens: list[int] = []

    best_score = -1e9
    global_step = 0
    num_updates = int(ppo_cfg.total_steps // (ppo_cfg.rollout_steps * ppo_cfg.n_envs))

    for update in range(num_updates):
        obs_np, _infos = venv.reset()
        obs = torch.as_tensor(obs_np, device=device, dtype=torch.float32)
        ac.reset_all_states()
        ep_returns[:] = 0.0
        ep_lens[:] = 0

        buf = RolloutBufferContinuous.allocate(
            T=ppo_cfg.rollout_steps, N=ppo_cfg.n_envs, obs_dim=obs_dim, action_dim=action_dim, device=device
        )

        for _step in range(ppo_cfg.rollout_steps):
            global_step += ppo_cfg.n_envs
            with torch.no_grad():
                mean, log_std, values, spike_rate = ac(obs)
                log_std = torch.clamp(log_std, -5.0, 2.0)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                u = dist.sample()
                actions, logprobs = _tanh_squash_action(dist, u)

            vec_step = venv.step(actions.detach().cpu().numpy().reshape(-1))
            next_obs = torch.as_tensor(vec_step.obs, device=device, dtype=torch.float32)
            rewards = torch.as_tensor(vec_step.rewards, device=device, dtype=torch.float32)
            dones = torch.as_tensor(vec_step.dones, device=device, dtype=torch.bool)

            ep_returns += vec_step.rewards
            ep_lens += 1
            for i, d in enumerate(vec_step.dones.tolist()):
                if d:
                    completed_returns.append(float(ep_returns[i]))
                    completed_lens.append(int(ep_lens[i]))
                    ep_returns[i] = 0.0
                    ep_lens[i] = 0

            buf.add(obs=obs, actions=actions, logprobs=logprobs, rewards=rewards, dones=dones, values=values)

            ac.reset_done(dones)
            obs = next_obs

        with torch.no_grad():
            _m, _ls, last_values, _sr = ac(obs)

        buf.compute_returns_advantages(last_values=last_values, gamma=ppo_cfg.gamma, gae_lambda=ppo_cfg.gae_lambda)
        advantages = buf.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        clip_fracs = []
        spike_mean_val = 0.0
        homeo_loss_val = 0.0
        for epoch in range(ppo_cfg.update_epochs):
            ac.reset_all_states()

            pg_loss_sum = torch.tensor(0.0, device=device)
            value_loss_sum = torch.tensor(0.0, device=device)
            entropy_sum = torch.tensor(0.0, device=device)
            spike_sum = torch.tensor(0.0, device=device)

            for t in range(ppo_cfg.rollout_steps):
                mean, log_std, new_values, spike_rate = ac(buf.obs[t])
                log_std = torch.clamp(log_std, -5.0, 2.0)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)

                u = _atanh(buf.actions[t])
                new_logprob_u = dist.log_prob(u)
                log_det = torch.log(1.0 - buf.actions[t].pow(2) + 1e-6)
                new_logprob = (new_logprob_u - log_det).sum(dim=-1)

                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = (new_logprob - buf.logprobs[t]).exp()
                adv_t = advantages[t]

                pg_loss1 = -adv_t * ratio
                pg_loss2 = -adv_t * torch.clamp(ratio, 1.0 - ppo_cfg.clip_coef, 1.0 + ppo_cfg.clip_coef)
                pg_loss_t = torch.max(pg_loss1, pg_loss2).mean()

                value_loss_t = 0.5 * (buf.returns[t] - new_values).pow(2).mean()

                pg_loss_sum = pg_loss_sum + pg_loss_t
                value_loss_sum = value_loss_sum + value_loss_t
                entropy_sum = entropy_sum + entropy
                spike_sum = spike_sum + spike_rate
                clip_fracs.append(((ratio - 1.0).abs() > ppo_cfg.clip_coef).float().mean().item())

                ac.reset_done(buf.dones[t])

            pg_loss = pg_loss_sum / float(ppo_cfg.rollout_steps)
            value_loss = value_loss_sum / float(ppo_cfg.rollout_steps)
            entropy_mean = entropy_sum / float(ppo_cfg.rollout_steps)
            spike_mean = spike_sum / float(ppo_cfg.rollout_steps)
            homeo_loss = spike_homeostasis_loss(spike_mean, target_rate=homeo_target, coef=homeo_coef)

            loss = (
                pg_loss
                + ppo_cfg.vf_coef * value_loss
                - ppo_cfg.ent_coef * entropy_mean
                + ppo_cfg.spike_rate_coef * spike_mean
                + homeo_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), ppo_cfg.max_grad_norm)
            optimizer.step()
            spike_mean_val = float(spike_mean.detach().cpu().item())
            homeo_loss_val = float(homeo_loss.detach().cpu().item())

        ep_ret_mean = float(np.mean(completed_returns[-100:])) if completed_returns else float("nan")
        ep_len_mean = float(np.mean(completed_lens[-100:])) if completed_lens else float("nan")
        logger.log(
            {
                "update": update,
                "global_step": global_step,
                "ep_return_mean_100": ep_ret_mean,
                "ep_len_mean_100": ep_len_mean,
                "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
                "spike_mean": float(spike_mean_val),
                "homeo_loss": float(homeo_loss_val),
            }
        )

        score = ep_ret_mean if np.isfinite(ep_ret_mean) else -1e9
        if score > best_score:
            best_score = score
            torch.save(ac.state_dict(), str(ckpt_dir / "best.pt"))

    torch.save(ac.state_dict(), str(ckpt_dir / "last.pt"))
    return {"best_score": float(best_score), "global_step": int(global_step)}
