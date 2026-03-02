from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bci_snn_rl.models.actor_critic import ActorCritic, ActorCriticConfig  # noqa: E402


def test_actor_critic_reset_restores_encoder_state() -> None:
    torch.manual_seed(0)
    cfg = ActorCriticConfig(
        encoder_type="sigma_delta",
        encoder_learnable_threshold=False,
        # Pick theta so the first call emits spikes and the second does not.
        encoder_theta_init=1.0,
        actor_type="ann_mlp",
        actor_hidden_sizes=[8],
        critic_hidden_sizes=[8],
    )
    ac = ActorCritic(obs_dim=4, action_dim=3, cfg=cfg)
    obs = torch.full((1, 4), 0.7, dtype=torch.float32)

    logits1, _v1, _sr1 = ac(obs)
    logits2, _v2, _sr2 = ac(obs)  # encoder has state, so this should differ
    assert not torch.allclose(logits1, logits2)

    ac.reset_done(torch.tensor([True]))
    logits3, _v3, _sr3 = ac(obs)
    assert torch.allclose(logits1, logits3)
