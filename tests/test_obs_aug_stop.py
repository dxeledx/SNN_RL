from __future__ import annotations

import numpy as np

from bci_snn_rl.data.obs_aug import augment_stop_windows


def test_augment_stop_windows_subject_onehot_and_time() -> None:
    N, S, D = 2, 4, 3
    X = np.zeros((N, S, D), dtype=np.float32)
    subjects = [1, 2, 3]

    X_aug = augment_stop_windows(
        X,
        subject=2,
        subjects=subjects,
        add_subject_onehot=True,
        add_time_feature=True,
    )

    assert X_aug.shape == (N, S, D + len(subjects) + 1)

    onehot = X_aug[:, :, D : D + len(subjects)]
    assert np.all(onehot[:, :, 1] == 1.0)
    assert np.all(onehot[:, :, 0] == 0.0)
    assert np.all(onehot[:, :, 2] == 0.0)

    t_norm = X_aug[:, :, -1]
    assert np.allclose(t_norm[:, 0], 0.0)
    assert np.allclose(t_norm[:, -1], 1.0)

