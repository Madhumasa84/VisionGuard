"""
VisionGuard Lite - Comprehensive Evaluation Metrics
Calculates:
- Precision, Recall, F1-score
- AUROC, AUPRC
- Brier Score
- Confusion Matrix
- Latency profiling (median & P95 latency in ms with warm-up)
- Serialized model size
"""

import time
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)
from typing import Dict, Any, Tuple

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    y_true: 1D array of ground truth {0, 1}
    y_pred: 1D array of predicted class {0, 1}
    y_prob: 1D array of predicted probability for class 1 (Fall)
    """
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = 0.0

    brier = float(brier_score_loss(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'auroc': auroc,
        'auprc': auprc,
        'brier_score': brier,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }

def profile_model_inference(
    model: nn.Module,
    input_sample: torch.Tensor,
    device: torch.device,
    warmup_runs: int = 20,
    benchmark_runs: int = 100
) -> Dict[str, float]:
    """
    Profiles single-sample (batch size = 1) inference latency in milliseconds.
    """
    model.eval()
    model.to(device)
    x = input_sample.to(device)
    if x.dim() == 2:
        x = x[0:1] # Ensure batch size 1
    elif x.dim() == 3:
        x = x[0:1]

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0) # convert to ms

    times = np.array(times)
    return {
        'median_latency_ms': float(np.median(times)),
        'p95_latency_ms': float(np.percentile(times, 95)),
        'mean_latency_ms': float(np.mean(times)),
        'std_latency_ms': float(np.std(times))
    }

def get_serialized_model_size_kb(checkpoint_path: str) -> float:
    if os.path.exists(checkpoint_path):
        return round(os.path.getsize(checkpoint_path) / 1024.0, 2)
    return 0.0
