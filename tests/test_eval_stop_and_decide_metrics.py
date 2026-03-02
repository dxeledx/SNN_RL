from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bci_snn_rl.eval.evaluate import eval_stop_and_decide  # noqa: E402
from bci_snn_rl.models.actor_critic import ActorCritic, ActorCriticConfig  # noqa: E402


def test_eval_stop_and_decide_commit_rate_and_action_fracs(tmp_path: Path) -> None:
    # Make a checkpoint that always commits LEFT at the first step.
    cfg_ac = ActorCriticConfig(
        encoder_type="none",
        encoder_learnable_threshold=False,
        encoder_theta_init=0.25,
        actor_type="ann_mlp",
        actor_hidden_sizes=[8],
        critic_hidden_sizes=[8],
    )
    ac = ActorCritic(obs_dim=4, action_dim=3, cfg=cfg_ac)
    with torch.no_grad():
        ac.actor_head.weight.zero_()
        ac.actor_head.bias[:] = torch.tensor([0.0, 10.0, 0.0], dtype=ac.actor_head.bias.dtype)

    ckpt = tmp_path / "ckpt.pt"
    torch.save(ac.state_dict(), ckpt)

    # 5 trials, 3 steps, 4-dim obs; label is always 0 so commit-left is always correct.
    X_eval = np.zeros((5, 3, 4), dtype=np.float32)
    y_eval = np.zeros((5,), dtype=np.int64)

    cfg = {"task": {"time_cost": 0.0, "no_commit_penalty": -1.0}, "model": {"encoder": {"type": "none"}, "actor": {"type": "ann_mlp", "hidden_sizes": [8]}, "critic": {"hidden_sizes": [8]}, "snn": {"input_scale": 1.0, "v_threshold": 1.0}}}

    summary = eval_stop_and_decide(
        cfg=cfg,
        X_eval=X_eval,
        y_eval=y_eval,
        checkpoint_path=str(ckpt),
        device=torch.device("cpu"),
    )

    assert summary.n_trials == 5
    assert summary.acc == 1.0
    assert summary.mdt_steps_mean == 1.0
    assert summary.commit_rate == 1.0
    assert summary.mean_commit_step == 1.0
    assert summary.action_frac_continue == 0.0
    assert summary.action_frac_commit_left == 1.0
    assert summary.action_frac_commit_right == 0.0

