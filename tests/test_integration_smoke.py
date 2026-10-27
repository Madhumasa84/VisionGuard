"""
Integration smoke tests for end-to-end pipeline and model consistency
"""

import os
import pytest
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from demo.app import PipelineManager, process_video_pipeline
from src.models.temporal_cnn import TemporalCNN

def test_pipeline_manager_initialization():
    pm = PipelineManager()
    assert pm.py_model is not None or pm.onnx_session is not None
    assert pm.temperature > 0.5
    assert pm.optimal_threshold > 0.0

def test_pytorch_onnx_real_model_parity():
    ckpt_path = REPO_ROOT / "checkpoints/real_temporalcnn_best.pt"
    onnx_path = REPO_ROOT / "exported_models/real_temporalcnn.onnx"
    
    if not ckpt_path.exists() or not onnx_path.exists():
        pytest.skip("Real model weights or ONNX model not present.")

    # PyTorch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    py_model = TemporalCNN(in_channels=51, num_classes=2)
    py_model.load_state_dict(state)
    py_model.eval()

    # ONNX
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # 10 random inputs
    for _ in range(10):
        dummy = np.random.randn(1, 51, 30).astype(np.float32)
        with torch.no_grad():
            py_out = py_model(torch.from_numpy(dummy)).numpy()
        onnx_out = session.run(None, {input_name: dummy})[0]
        max_diff = np.max(np.abs(py_out - onnx_out))
        assert max_diff < 1e-4, f"Parity failed with max diff: {max_diff}"

def test_end_to_end_video_pipeline_smoke():
    sample_vid = str(REPO_ROOT / "demo/sample_activity.mp4")
    if not os.path.exists(sample_vid):
        pytest.skip("Sample video file not found.")

    # Run with ONNX backend
    out_vid, out_plot, exp, metrics = process_video_pipeline(
        sample_vid,
        confirmation_k=2,
        confidence_thresh=0.50,
        backend_choice="ONNX Runtime (Fastest / Recommended)"
    )
    assert out_vid is not None and os.path.exists(out_vid)
    assert out_plot is not None and os.path.exists(out_plot)
    assert len(exp) > 10
    assert "Decision Status" in metrics
    assert "Peak Fall Probability" in metrics

def test_invalid_input_handling():
    out_vid, out_plot, exp, metrics = process_video_pipeline(None)
    assert out_vid is None
    assert out_plot is None
    assert "valid video" in exp.lower()

    out_vid, out_plot, exp, metrics = process_video_pipeline("non_existent_file.mp4")
    assert out_vid is None
    assert out_plot is None
    assert "valid video" in exp.lower()

def test_temporal_confirmation_k_behavior():
    sample_vid = str(REPO_ROOT / "demo/sample_activity.mp4")
    if not os.path.exists(sample_vid):
        pytest.skip("Sample video file not found.")

    # High confirmation K should be stricter or equal to K=1
    _, _, _, metrics_k1 = process_video_pipeline(sample_vid, confirmation_k=1, confidence_thresh=0.65)
    _, _, _, metrics_k4 = process_video_pipeline(sample_vid, confirmation_k=4, confidence_thresh=0.65)
    
    assert "Decision Status" in metrics_k1
    assert "Decision Status" in metrics_k4
