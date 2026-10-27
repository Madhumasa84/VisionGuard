"""
Unit tests for calibration and uncertainty estimation
"""

import pytest
import numpy as np
import torch
from src.evaluation.uncertainty import TemperatureScaler, compute_ece, UncertaintyPolicy

def test_temperature_scaler():
    scaler = TemperatureScaler()
    logits = np.array([[2.5, -2.5], [-1.0, 3.0], [0.5, 0.2]], dtype=np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)
    opt_t = scaler.fit(logits, labels, max_iter=20)
    assert opt_t > 0.05
    scaled = scaler(torch.tensor(logits)).detach().numpy()
    assert scaled.shape == (3, 2)

def test_compute_ece():
    # Perfectly calibrated case
    probs = np.array([
        [0.8, 0.2],
        [0.8, 0.2],
        [0.8, 0.2],
        [0.8, 0.2],
        [0.8, 0.2]
    ])
    labels = np.array([0, 0, 0, 0, 1]) # 4 out of 5 correct = 80% accuracy == 80% confidence
    ece_res = compute_ece(probs, labels, n_bins=5)
    assert 'ece' in ece_res
    assert ece_res['ece'] < 0.05

def test_uncertainty_policy():
    policy = UncertaintyPolicy(confidence_threshold=0.80, max_entropy=0.50)
    # Ambiguous probability
    ambig_prob = np.array([[0.55, 0.45]])
    decisions = policy.predict_with_rejection(ambig_prob)
    assert decisions[0]['decision'] == 'UNCERTAIN'
    assert decisions[0]['is_uncertain'] is True

    # High confidence fall
    fall_prob = np.array([[0.02, 0.98]])
    decisions_fall = policy.predict_with_rejection(fall_prob)
    assert decisions_fall[0]['decision'] == 'FALL'
    assert decisions_fall[0]['is_uncertain'] is False
