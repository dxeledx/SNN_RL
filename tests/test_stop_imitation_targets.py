import numpy as np

from bci_snn_rl.run_pretrain_stop_imitation import compute_commit_steps_from_probs, make_stop_imitation_targets


def test_commit_step_first_crossing_or_last() -> None:
    # 2 trials, 5 steps.
    probs = np.zeros((2, 5, 2), dtype=np.float32)

    # trial0: crosses at step2
    # keep early steps near 0.5 (low confidence), then cross threshold.
    probs[0, :, 0] = [0.55, 0.52, 0.7, 0.8, 0.9]
    probs[0, :, 1] = 1.0 - probs[0, :, 0]

    # trial1: never crosses -> last step
    probs[1, :, 0] = [0.55, 0.52, 0.51, 0.56, 0.59]
    probs[1, :, 1] = 1.0 - probs[1, :, 0]

    cs = compute_commit_steps_from_probs(probs, threshold=0.6)
    assert cs.tolist() == [2, 4]


def test_make_targets_continue_then_commit_then_ignore() -> None:
    y = np.array([0, 1], dtype=np.int64)
    commit_steps = np.array([2, 0], dtype=np.int64)
    targets = make_stop_imitation_targets(y=y, commit_steps=commit_steps, n_steps=5, ignore_index=-100)

    # trial0: continue, continue, commit_left, ignore, ignore
    assert targets[0].tolist() == [0, 0, 1, -100, -100]

    # trial1: commit_right at t=0, rest ignored
    assert targets[1].tolist() == [2, -100, -100, -100, -100]
