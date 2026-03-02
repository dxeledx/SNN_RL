from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bci_snn_rl.rl.ppo import spike_homeostasis_loss  # noqa: E402


def test_spike_homeostasis_loss_is_zero_at_target_and_monotone() -> None:
    target = 0.05
    coef = 2.0

    loss_at_target = spike_homeostasis_loss(torch.tensor(target), target_rate=target, coef=coef)
    assert float(loss_at_target.detach().cpu().item()) == pytest.approx(0.0)

    loss_high = spike_homeostasis_loss(torch.tensor(0.10), target_rate=target, coef=coef)
    loss_low = spike_homeostasis_loss(torch.tensor(0.00), target_rate=target, coef=coef)
    assert float(loss_high.detach().cpu().item()) > 0.0
    assert float(loss_low.detach().cpu().item()) > 0.0
    # Symmetric around target for squared error.
    assert float(loss_high.detach().cpu().item()) == pytest.approx(float(loss_low.detach().cpu().item()))


def test_spike_homeostasis_loss_can_be_disabled() -> None:
    loss = spike_homeostasis_loss(torch.tensor(0.2), target_rate=0.05, coef=0.0)
    assert float(loss.detach().cpu().item()) == pytest.approx(0.0)

