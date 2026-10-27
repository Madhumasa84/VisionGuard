"""
Unit tests for SLM explainer, safety validation, and ONNX parity
"""

import os
import pytest
import numpy as np
import onnxruntime as ort
import torch

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.inference.slm_explainer import TemplateExplainer, EvidenceConsistencyChecker

def test_template_explainer_fall():
    explainer = TemplateExplainer()
    evidence = {
        'detected_activity': 'fall',
        'confidence': 0.95,
        'landmark_visibility': 0.85,
        'temporal_consistency': 'high',
        'uncertain': False
    }
    res = explainer.explain(evidence)
    assert 'ALERT' in res['explanation'] or 'fall' in res['explanation'].lower()
    
    check = EvidenceConsistencyChecker.validate(evidence, res['explanation'])
    assert check['passed'] is True

def test_template_explainer_uncertain():
    explainer = TemplateExplainer()
    evidence = {
        'detected_activity': 'possible_fall',
        'confidence': 0.55,
        'landmark_visibility': 0.40,
        'temporal_consistency': 'mixed',
        'uncertain': True
    }
    res = explainer.explain(evidence)
    assert 'Uncertain' in res['explanation']
    check = EvidenceConsistencyChecker.validate(evidence, res['explanation'])
    assert check['passed'] is True

def test_evidence_consistency_checker_catches_hallucination():
    evidence = {'detected_activity': 'fall', 'confidence': 0.90, 'uncertain': False}
    # Unverified medical hallucination
    bad_explanation = "The patient suffered a severe bone fracture and paramedics were dispatched."
    check = EvidenceConsistencyChecker.validate(evidence, bad_explanation)
    assert check['passed'] is False
    assert any('fracture' in v or 'paramedics' in v for v in check['violations'])

def test_onnx_model_inference():
    onnx_path = str(REPO_ROOT / 'exported_models/temporal_cnn_clean.onnx')
    if not os.path.exists(onnx_path):
        onnx_path = str(REPO_ROOT / 'exported_models/temporal_cnn.onnx')
    if not os.path.exists(onnx_path):
        pytest.skip("ONNX model file not found yet")
        
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 51, 30).astype(np.float32)
    out = session.run(None, {input_name: dummy})[0]
    assert out.shape == (1, 2)
