"""
Unit tests for data preprocessing and schemas
"""

import pytest
import numpy as np
from src.data.preprocessing import PosePreprocessor
from src.data.schema import WINDOW_SIZE, to_binary_label

def test_to_binary_label():
    assert to_binary_label(6) == 1
    assert to_binary_label(0) == 0
    assert to_binary_label(3) == 0

def test_pose_normalization():
    prep = PosePreprocessor(target_window_size=30)
    # Synthetic sequence (30, 17, 3)
    kps = np.random.uniform(0.2, 0.8, size=(30, 17, 3)).astype(np.float32)
    # Fix hips and shoulders
    kps[:, 11, :2] = [0.45, 0.5] # l_hip
    kps[:, 12, :2] = [0.55, 0.5] # r_hip
    kps[:, 5, :2] = [0.45, 0.2]  # l_sh
    kps[:, 6, :2] = [0.55, 0.2]  # r_sh

    norm = prep.normalize_skeleton(kps)
    assert norm.shape == (30, 17, 3)
    # After normalization, hip midpoint should be approximately (0, 0)
    hip_mid = (norm[:, 11, :2] + norm[:, 12, :2]) / 2.0
    assert np.allclose(hip_mid, 0.0, atol=1e-5)

def test_missing_landmark_imputation():
    prep = PosePreprocessor(target_window_size=30)
    kps = np.ones((20, 17, 3), dtype=np.float32)
    kps[:, :, 0] = np.arange(20)[:, None]
    # Mark frames 5-8 of joint 0 as missing (conf = 0.1)
    kps[5:9, 0, 2] = 0.1
    imputed = prep.impute_missing_landmarks(kps, min_conf=0.3)
    # Check that imputed frames linearly interpolate between frame 4 (val 4.0) and frame 9 (val 9.0)
    assert abs(imputed[5, 0, 0] - 5.0) < 1e-4
    assert abs(imputed[7, 0, 0] - 7.0) < 1e-4

def test_sliding_windows_fall_labeling():
    prep = PosePreprocessor(target_window_size=30)
    kps = np.random.randn(100, 17, 3).astype(np.float32)
    fall_intervals = [(30, 60)] # Fall between frames 30 and 60

    windows, labels, meta = prep.create_sliding_windows(
        kps, stride=10, fall_intervals=fall_intervals, is_fall_video=True
    )
    assert len(windows) == len(labels)
    # Window starting at frame 0 (0-30) should be non-fall (0)
    assert labels[0] == 0
    # Window overlapping >= 50% (e.g. 30-60) should be 1 (fall)
    assert labels[3] == 1
