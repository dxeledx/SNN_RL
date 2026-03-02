from __future__ import annotations

import torch

from bci_snn_rl.utils.stop_policy import mask_continue_at_last_step


def test_mask_continue_at_last_step() -> None:
    # logits prefer CONTINUE for both items
    logits = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    obs = torch.tensor([[0.2, 0.0], [0.2, 1.0]])  # last dim is t_norm

    masked = mask_continue_at_last_step(logits, obs)

    # First item is not last step.
    assert float(masked[0, 0].item()) == 10.0

    # Second item is last step -> CONTINUE is masked out.
    assert float(masked[1, 0].item()) < -1e8
    assert int(torch.argmax(masked[1]).item()) != 0

    # Original logits must not be modified in-place.
    assert float(logits[1, 0].item()) == 10.0

