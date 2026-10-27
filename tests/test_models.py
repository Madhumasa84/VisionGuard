"""
Unit tests for model architectures
"""

import pytest
import torch
from src.models.baseline_mlp import BaselineMLP
from src.models.temporal_cnn import TemporalCNN
from src.models.small_tcn import SmallTCN

def test_baseline_mlp_forward():
    model = BaselineMLP(in_features=1530, num_classes=2)
    x = torch.randn(8, 1530)
    out = model(x)
    assert out.shape == (8, 2)
    assert model.count_parameters() > 1000

def test_temporal_cnn_forward():
    model = TemporalCNN(in_channels=51, num_classes=2)
    x = torch.randn(8, 51, 30)
    out = model(x)
    assert out.shape == (8, 2)
    # Also test 4D input (B, T, J, C)
    x4d = torch.randn(8, 30, 17, 3)
    out4d = model(x4d)
    assert out4d.shape == (8, 2)

def test_small_tcn_forward():
    model = SmallTCN(in_channels=51, num_classes=2)
    x = torch.randn(8, 51, 30)
    out = model(x)
    assert out.shape == (8, 2)
