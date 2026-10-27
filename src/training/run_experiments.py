"""
VisionGuard Lite - Autonomous Experiment Runner & Benchmark Suite
Trains Model A, B, C; computes test metrics on held-out subjects (P11, P12);
fits temperature scaling calibration; evaluates distribution shifts;
profiles inference latency; exports ONNX; and generates research plots.
"""

import os
import sys
sys.path.insert(0, '/content/drive/MyDrive/VisionGuardLite')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from src.data.dataset import get_dataloaders
from src.models.baseline_mlp import BaselineMLP
from src.models.temporal_cnn import TemporalCNN
from src.models.small_tcn import SmallTCN
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import profile_model_inference, get_serialized_model_size_kb
from src.evaluation.uncertainty import (
    TemperatureScaler,
    compute_ece,
    UncertaintyPolicy,
    evaluate_distribution_shifts
)
from src.inference.onnx_exporter import (
    export_temporal_cnn_to_onnx,
    benchmark_onnx_latency
)

def run_all_experiments(
    project_dir: str = '/content/drive/MyDrive/VisionGuardLite',
    data_dir: str = '/content/drive/MyDrive/VisionGuardLite/datasets/processed/coco_17'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing experiments on device: {device}")

    checkpoints_dir = os.path.join(project_dir, 'checkpoints')
    results_metrics_dir = os.path.join(project_dir, 'results/metrics')
    results_plots_dir = os.path.join(project_dir, 'results/plots')
    exported_dir = os.path.join(project_dir, 'exported_models')

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_metrics_dir, exist_ok=True)
    os.makedirs(results_plots_dir, exist_ok=True)
    os.makedirs(exported_dir, exist_ok=True)

    # 1. Load Data
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        data_dir=data_dir, batch_size=32, flatten_for_mlp=False
    )
    # Also get flattened loaders for Model A
    train_loader_mlp, val_loader_mlp, test_loader_mlp, _ = get_dataloaders(
        data_dir=data_dir, batch_size=32, flatten_for_mlp=True
    )

    models_config = [
        ('Model A (BaselineMLP)', BaselineMLP(in_features=1530), train_loader_mlp, val_loader_mlp, test_loader_mlp, 'baseline_mlp'),
        ('Model B (TemporalCNN)', TemporalCNN(in_channels=51), train_loader, val_loader, test_loader, 'temporal_cnn'),
        ('Model C (SmallTCN)', SmallTCN(in_channels=51), train_loader, val_loader, test_loader, 'small_tcn')
    ]

    all_results = {}
    saved_histories = {}
    trained_models = {}
    val_logits_dict = {}
    test_logits_dict = {}

    plt.figure(figsize=(12, 4))

    for model_title, model_inst, tr_l, val_l, te_l, ckpt_name in models_config:
        print(f"\n==========================================")
        print(f" Training: {model_title}")
        print(f"==========================================")
        ckpt_path = os.path.join(checkpoints_dir, f"{ckpt_name}_best.pt")

        trainer = ModelTrainer(
            model=model_inst,
            device=device,
            learning_rate=1e-3,
            class_weights=class_weights,
            max_epochs=25,
            patience=6,
            checkpoint_path=ckpt_path
        )

        history = trainer.fit(tr_l, val_l)
        saved_histories[ckpt_name] = history

        # Evaluate on validation
        val_eval = trainer.evaluate(val_l)
        val_logits_dict[ckpt_name] = val_eval['logits']

        # Evaluate on held-out test subjects (P11, P12)
        test_eval = trainer.evaluate(te_l)
        test_logits_dict[ckpt_name] = test_eval

        # Latency profiling
        sample_x, _ = next(iter(te_l))
        latency_info = profile_model_inference(model_inst, sample_x, device=device, benchmark_runs=100)

        # Model size
        size_kb = get_serialized_model_size_kb(ckpt_path)

        all_results[model_title] = {
            'parameters': model_inst.count_parameters(),
            'model_size_kb': size_kb,
            'precision': round(test_eval['precision'], 4),
            'recall': round(test_eval['recall'], 4),
            'f1_score': round(test_eval['f1_score'], 4),
            'auroc': round(test_eval['auroc'], 4),
            'auprc': round(test_eval['auprc'], 4),
            'brier_score': round(test_eval['brier_score'], 4),
            'median_latency_ms': round(latency_info['median_latency_ms'], 3),
            'p95_latency_ms': round(latency_info['p95_latency_ms'], 3),
            'confusion_matrix': test_eval['confusion_matrix']
        }
        trained_models[ckpt_name] = model_inst

    # Plot Training Loss Curves
    plt.subplot(1, 2, 1)
    for name, hist in saved_histories.items():
        plt.plot(hist['train_loss'], label=f"{name} train")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for name, hist in saved_histories.items():
        plt.plot(hist['val_loss'], label=f"{name} val")
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_plots_dir, 'training_loss_curves.png'), dpi=150)
    plt.close()

    # Save Model Comparison JSON
    comparison_file = os.path.join(results_metrics_dir, 'model_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved model comparison table to {comparison_file}")

    # 2. Probability Calibration via Temperature Scaling (on Model B)
    print("\nFitting Temperature Scaler for Model B (TemporalCNN)...")
    val_y = np.load(os.path.join(data_dir, 'y_val.npy'))
    test_y = np.load(os.path.join(data_dir, 'y_test.npy'))

    scaler = TemperatureScaler()
    opt_temp = scaler.fit(val_logits_dict['temporal_cnn'], val_y)

    uncal_logits = test_logits_dict['temporal_cnn']['logits']
    uncal_probs = torch.softmax(torch.tensor(uncal_logits), dim=1).numpy()
    cal_logits = uncal_logits / opt_temp
    cal_probs = torch.softmax(torch.tensor(cal_logits), dim=1).numpy()

    ece_before = compute_ece(uncal_probs, test_y, n_bins=10)
    ece_after = compute_ece(cal_probs, test_y, n_bins=10)

    calibration_results = {
        'optimal_temperature': round(opt_temp, 4),
        'test_ece_before': round(ece_before['ece'], 4),
        'test_ece_after': round(ece_after['ece'], 4),
        'improvement_factor': round(ece_before['ece'] / max(1e-5, ece_after['ece']), 2),
        'bin_stats_before': ece_before['bin_stats'],
        'bin_stats_after': ece_after['bin_stats']
    }
    with open(os.path.join(results_metrics_dir, 'calibration_results.json'), 'w') as f:
        json.dump(calibration_results, f, indent=2)

    # Plot Reliability Diagram
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    if len(ece_before['bin_stats']) > 0:
        c_b = [b['confidence'] for b in ece_before['bin_stats']]
        a_b = [b['accuracy'] for b in ece_before['bin_stats']]
        plt.plot(c_b, a_b, 's-', color='red', label=f"Uncalibrated (ECE={ece_before['ece']:.3f})")
    if len(ece_after['bin_stats']) > 0:
        c_a = [b['confidence'] for b in ece_after['bin_stats']]
        a_a = [b['accuracy'] for b in ece_after['bin_stats']]
        plt.plot(c_a, a_a, 'o-', color='green', label=f"Calibrated (T={opt_temp:.3f}, ECE={ece_after['ece']:.3f})")
    plt.title("Reliability Diagram (Model B on Test Set)")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_plots_dir, 'reliability_diagram.png'), dpi=150)
    plt.close()

    # 3. Distribution Shift Resilience Stress-Testing
    print("\nStress-testing Model B under domain shifts (occlusions, noise, scaling)...")
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    shift_results = evaluate_distribution_shifts(trained_models['temporal_cnn'], X_test, test_y, device=device)
    with open(os.path.join(results_metrics_dir, 'distribution_shift_results.json'), 'w') as f:
        json.dump(shift_results, f, indent=2)

    # Plot Distribution Shift Degradation
    shifts = list(shift_results.keys())
    f1s = [shift_results[s]['f1_score'] for s in shifts]
    aurocs = [shift_results[s]['auroc'] for s in shifts]
    x_idx = np.arange(len(shifts))

    plt.figure(figsize=(8, 4))
    plt.bar(x_idx - 0.15, f1s, width=0.3, label='F1-Score', color='royalblue')
    plt.bar(x_idx + 0.15, aurocs, width=0.3, label='AUROC', color='darkorange')
    plt.xticks(x_idx, [s.replace('_', '\n') for s in shifts])
    plt.ylim(0, 1.1)
    plt.title("Model B Performance Under Sensor Degradation")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_plots_dir, 'distribution_shift_degradation.png'), dpi=150)
    plt.close()

    # 4. ONNX Export and Parity Verification
    print("\nExporting Model B to ONNX...")
    onnx_path = os.path.join(exported_dir, 'temporal_cnn.onnx')
    onnx_info = export_temporal_cnn_to_onnx(
        model=trained_models['temporal_cnn'],
        checkpoint_path=os.path.join(checkpoints_dir, 'temporal_cnn_best.pt'),
        output_onnx_path=onnx_path
    )
    bench_info = benchmark_onnx_latency(onnx_path)
    onnx_info.update(bench_info)
    with open(os.path.join(results_metrics_dir, 'onnx_benchmark.json'), 'w') as f:
        json.dump(onnx_info, f, indent=2)
    print(f"ONNX Model exported ({onnx_info['model_size_kb']} KB), Parity diff: {onnx_info['max_absolute_diff']:.6f}")

    print("\nAll experiments, calibration, stress tests, and ONNX exports completed successfully!")
    return {
        'model_comparison': all_results,
        'calibration': calibration_results,
        'shifts': shift_results,
        'onnx': onnx_info
    }

if __name__ == '__main__':
    run_all_experiments()

