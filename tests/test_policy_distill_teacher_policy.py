from __future__ import annotations

import numpy as np

from bci_snn_rl.run_pretrain_stop_lda_policy_distill import _teacher_policy_from_probs


def test_teacher_policy_from_probs_is_normalized_and_monotone() -> None:
    probs = np.array([[[0.9, 0.1]], [[0.5, 0.5]]], dtype=np.float32)  # [N=2,S=1,2]
    pi = _teacher_policy_from_probs(probs)
    assert pi.shape == (2, 1, 3)

    # In [0,1] and rows sum to 1.
    assert np.all(pi >= 0.0)
    assert np.all(pi <= 1.0)
    assert np.allclose(pi.sum(axis=-1), 1.0)

    # Higher confidence => lower continue probability.
    continue_high_conf = float(pi[0, 0, 0])
    continue_low_conf = float(pi[1, 0, 0])
    assert continue_high_conf < continue_low_conf

