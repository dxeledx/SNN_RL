from __future__ import annotations

import torch

from bci_snn_rl.envs.stop_and_decide import ACTION_CONTINUE


def mask_continue_at_last_step(logits: torch.Tensor, obs: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """
    Force-commit at the last step by masking out ACTION_CONTINUE.

    Convention: obs[..., -1] is the time feature t_norm in [0,1], and the last step is t_norm==1.0.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits [N,A], got {tuple(logits.shape)}")
    if obs.ndim != 2:
        raise ValueError(f"Expected obs [N,D], got {tuple(obs.shape)}")
    if logits.shape[0] != obs.shape[0]:
        raise ValueError(f"Batch size mismatch: logits {tuple(logits.shape)} vs obs {tuple(obs.shape)}")
    if obs.shape[1] < 1:
        raise ValueError("obs must have at least 1 feature (time feature at last dim)")

    is_last = obs[:, -1] >= (1.0 - float(eps))
    if not torch.any(is_last):
        return logits

    masked = logits.clone()
    masked[is_last, ACTION_CONTINUE] = -1e9
    return masked

