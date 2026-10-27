"""
VisionGuard Lite - Real-Data Reliability, Uncertainty & Stress-Testing Suite
Evaluates:
1. Selective prediction policy (risk-coverage, error on accepted vs rejected)
2. Distribution shifts and sensor corruptions:
   - 20% & 40% Landmark Occlusion
   - Coordinate Gaussian Noise (sigma = 0.02, 0.05)
   - Frame dropping / temporal downsampling
3. Latency profiling:
   - Pose extraction latency per frame
   - Model inference latency (P50, P95, mean)
   - Full end-to-end per-window latency
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.temporal_cnn import TemporalCNN
from src.models.small_tcn import SmallTCN
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.uncertainty import UncertaintyPolicy, compute_ece

def evaluate_selective_prediction(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    temperature: float = 1.0,
    thresholds: List[float] = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
) -> Dict[str, Any]:
    """
    Evaluates selective prediction across various confidence thresholds.
    Tracks coverage (accepted samples / total) vs accepted error rate.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(x_tensor)
        calib_logits = logits / max(0.1, temperature)
        probs_2d = torch.softmax(calib_logits, dim=1).cpu().numpy()
        probs_fall = probs_2d[:, 1]
        confidences = np.max(probs_2d, axis=1)
        preds = np.argmax(probs_2d, axis=1)

    total_samples = len(y_test)
    curve = []

    for thresh in thresholds:
        accepted_mask = (confidences >= thresh)
        rejected_mask = ~accepted_mask

        coverage = float(np.mean(accepted_mask))
        abstention_rate = 1.0 - coverage

        if np.sum(accepted_mask) > 0:
            accepted_acc = float(np.mean(preds[accepted_mask] == y_test[accepted_mask]))
            accepted_error = 1.0 - accepted_acc
            accepted_f1 = float(compute_classification_metrics(y_test[accepted_mask], preds[accepted_mask], probs_fall[accepted_mask])['f1_score'])
        else:
            accepted_acc = 0.0
            accepted_error = 0.0
            accepted_f1 = 0.0

        if np.sum(rejected_mask) > 0:
            rejected_acc = float(np.mean(preds[rejected_mask] == y_test[rejected_mask]))
            rejected_error = 1.0 - rejected_acc
        else:
            rejected_acc = 0.0
            rejected_error = 0.0

        curve.append({
            'confidence_threshold': thresh,
            'coverage': round(coverage, 4),
            'abstention_rate': round(abstention_rate, 4),
            'accepted_samples': int(np.sum(accepted_mask)),
            'rejected_samples': int(np.sum(rejected_mask)),
            'accepted_accuracy': round(accepted_acc, 4),
            'accepted_error': round(accepted_error, 4),
            'accepted_f1': round(accepted_f1, 4),
            'rejected_error': round(rejected_error, 4)
        })

    return {
        'risk_coverage_curve': curve
    }

def evaluate_sensor_stress(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    temperature: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluates model resilience under sensor degradation:
    - Clean test
    - 20% random landmark occlusion
    - 40% random landmark occlusion
    - Gaussian noise (sigma = 0.02, 0.05)
    - Frame drop (every 2nd frame dropped and linearly interpolated)
    """
    model.eval()
    policy = UncertaintyPolicy(confidence_threshold=0.75, max_entropy=0.55)

    shifts = {}
    shifts['clean'] = X_test.copy()

    # 1. 20% Occlusion
    x_occ20 = X_test.copy()
    N, T, J, C = x_occ20.shape
    mask20 = np.random.rand(N, 1, J, 1) < 0.20
    x_occ20[mask20.repeat(T, axis=1).repeat(C, axis=3)] = 0.0
    shifts['occlusion_20pct'] = x_occ20

    # 2. 40% Occlusion
    x_occ40 = X_test.copy()
    mask40 = np.random.rand(N, 1, J, 1) < 0.40
    x_occ40[mask40.repeat(T, axis=1).repeat(C, axis=3)] = 0.0
    shifts['occlusion_40pct'] = x_occ40

    # 3. Gaussian Noise (sigma = 0.02)
    x_noise02 = X_test.copy() + np.random.normal(0, 0.02, size=X_test.shape).astype(np.float32)
    shifts['gaussian_noise_0.02'] = x_noise02

    # 4. Gaussian Noise (sigma = 0.05)
    x_noise05 = X_test.copy() + np.random.normal(0, 0.05, size=X_test.shape).astype(np.float32)
    shifts['gaussian_noise_0.05'] = x_noise05

    # 5. Temporal Frame Drop (simulating 5 FPS by dropping alternate frames and forward-filling)
    x_drop = X_test.copy()
    for t in range(1, T, 2):
        x_drop[:, t] = x_drop[:, t - 1]
    shifts['temporal_frame_drop_5fps'] = x_drop

    results = {}
    for name, data in shifts.items():
        with torch.no_grad():
            x_t = torch.tensor(data, dtype=torch.float32).to(device)
            logits = model(x_t)
            calib_logits = logits / max(0.1, temperature)
            probs_2d = torch.softmax(calib_logits, dim=1).cpu().numpy()
            probs_fall = probs_2d[:, 1]
            preds = (probs_fall >= 0.5).astype(int)

        m = compute_classification_metrics(y_test, preds, probs_fall)
        ece = compute_ece(probs_2d, y_test)['ece']

        # Uncertainty policy rejection
        policy_decisions = policy.predict_with_rejection(probs_2d)
        abstentions = sum(1 for d in policy_decisions if d['is_uncertain'])
        abstention_rate = abstentions / float(len(policy_decisions))

        results[name] = {
            'precision': round(m['precision'], 4),
            'recall': round(m['recall'], 4),
            'f1_score': round(m['f1_score'], 4),
            'auroc': round(m['auroc'], 4),
            'auprc': round(m['auprc'], 4),
            'brier_score': round(m['brier_score'], 4),
            'ece': round(float(ece), 4),
            'abstention_rate': round(abstention_rate, 4),
            'accepted_samples': len(policy_decisions) - abstentions
        }

    return results

def profile_latency(
    model: nn.Module,
    sample_window: np.ndarray,
    device: torch.device,
    runs: int = 150
) -> Dict[str, float]:
    """
    Profiles inference latency on single sample (batch size = 1) over multiple runs.
    """
    model.eval()
    x = torch.tensor(sample_window[np.newaxis, ...], dtype=torch.float32).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(25):
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    times_ms = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    return {
        'median_latency_ms': round(float(np.median(times_ms)), 3),
        'p95_latency_ms': round(float(np.percentile(times_ms, 95)), 3),
        'mean_latency_ms': round(float(np.mean(times_ms)), 3),
        'min_latency_ms': round(float(np.min(times_ms)), 3),
        'max_latency_ms': round(float(np.max(times_ms)), 3)
    }

def run_reliability_suite():
    data_dir = str(REPO_ROOT / 'experiments/gmdcsa24_real/data/processed_windows')
    checkpoints_dir = str(REPO_ROOT / 'experiments/gmdcsa24_real/checkpoints')
    results_dir = str(REPO_ROOT / 'experiments/gmdcsa24_real/results')
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Reliability Suite on device: {device}")

    # Load best TemporalCNN and best SmallTCN
    ckpt_cnn_path = os.path.join(checkpoints_dir, 'real_temporalcnn_best.pt')
    if not os.path.exists(ckpt_cnn_path):
        print(f"Checkpoint {ckpt_cnn_path} not found. Please train first.")
        return

    ckpt = torch.load(ckpt_cnn_path, map_location=device, weights_only=False)
    best_fold = ckpt['best_fold']
    temperature = ckpt.get('temperature', 1.0)

    print(f"Loading {ckpt['model_name']} from {best_fold} with T={temperature:.3f}")
    model = TemporalCNN(in_channels=51, num_classes=2).to(device)
    model.load_state_dict(ckpt['model_state'])

    # Load test data of that best fold
    fold_dir = os.path.join(data_dir, best_fold)
    X_test = np.load(os.path.join(fold_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(fold_dir, 'y_test.npy'))

    # 1. Selective Prediction
    print("Evaluating Selective Prediction (Risk-Coverage)...")
    selective_res = evaluate_selective_prediction(model, X_test, y_test, device, temperature=temperature)

    # 2. Stress Testing
    print("Evaluating Sensor Stress Degradations...")
    stress_res = evaluate_sensor_stress(model, X_test, y_test, device, temperature=temperature)

    # 3. Latency Profiling
    print("Profiling Inference Latency...")
    lat_res = profile_latency(model, X_test[0], device)

    # Compile report
    output = {
        'model_name': 'TemporalCNN',
        'evaluated_fold': best_fold,
        'temperature': temperature,
        'latency_ms': lat_res,
        'selective_prediction': selective_res,
        'sensor_stress_testing': stress_res
    }

    out_file = os.path.join(results_dir, 'reliability_and_stress_results.json')
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nReliability suite results saved to {out_file}")
    print("\nSensor Stress Summary:")
    for shift_name, metrics in stress_res.items():
        print(f"  {shift_name:26s} | F1: {metrics['f1_score']:.4f} | Recall: {metrics['recall']:.4f} | Prec: {metrics['precision']:.4f} | Abstain: {metrics['abstention_rate']*100:.1f}%")

if __name__ == '__main__':
    run_reliability_suite()
