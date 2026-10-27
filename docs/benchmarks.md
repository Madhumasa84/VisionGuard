# Benchmark results

These are previously recorded research measurements, not a fresh evaluation of the updated demo. The demo now gates confirmation on pose visibility; that safeguard is not included in the historical ablation below. No training or benchmark regeneration was performed during the interface update.

### Table 1: Raw Unconfirmed LOSO Cross-Validation on Real Human Data (GMDCSA-24, Threshold = 0.50)

Evaluates unconfirmed single-window classifications without temporal confirmation across 4 held-out participant folds:

| Model Architecture | Parameters | Window F1 (Mean ± Std) | Window Recall | Window Precision | AUROC | Event Recall (Raw) | Unconfirmed FA/hr | Mean Detection Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TemporalCNN (Model B)** | **66,274** | **0.7570 ± 0.1175** | **0.8148** | **0.7149** | **0.8821** | **95.7%** | **119.72** | **1.66s** |
| BaselineMLP (Model A) | 100,322 | 0.7831 ± 0.1038 | 0.8966 | 0.6986 | 0.8926 | 96.7% | 140.43 | 1.49s |
| SmallTCN (Model C) | 67,154 | 0.6889 ± 0.0841 | 0.7513 | 0.6523 | 0.8516 | 89.9% | 131.83 | 1.61s |

*Bundled research model (TemporalCNN fold 3 checkpoint)*: **100% Event Recall (21/21 falls detected)**, **0.9104 Window F1**, **0.9815 AUROC**, **1.81s raw delay**.

*Calibration Assessment*: Mean Expected Calibration Error (ECE) across folds was **63.3% before** and **59.3% after** temperature scaling ($T = 1.627$). Under severe cross-participant domain shift, post-hoc scaling provides only minor calibration benefits (4.0% absolute reduction), demonstrating that static probability thresholding alone is insufficient and directly motivating temporal confirmation filtering.

### Table 2: Temporal Confirmation Policy Ablation on Continuous Video Monitoring (GMDCSA-24)

Evaluates the trade-off between false-alarm suppression and physical detection delay when requiring $K$ consecutive positive windows:

| Confirmation Policy | Definition | Overall Event Recall | Detected Events | Mean Physical Delay | False Alarm Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **K = 1 (Raw / Unfiltered)** | Single positive window fires alarm | **96.1%** | 74 / 77 | **1.66 s** | 119.72 FA/hr (~1 every 30.1s) |
| **K = 2 (Standard Confirmation)** | 2 consecutive positive windows | **92.2%** | 71 / 77 | **2.18 s** | 95.00 FA/hr |
| **K = 3 (Conservative)** | 3 consecutive positive windows | **87.0%** | 67 / 77 | **2.67 s** | 87.53 FA/hr |
| **K = 4 (High Latency)** | 4 consecutive positive windows | **70.1%** | 54 / 77 | **3.14 s** | 83.80 FA/hr |

K=2 is the demo default. Even with confirmation, the recorded false-alarm rate remains high. These results do not support unsupervised safety monitoring. Full fold-by-fold results are available in [confirmation_evaluation.json](../experiments/gmdcsa24_real/results/confirmation_evaluation.json).

### Table 3: Measured Consumer CPU Latency Profile (Batch Size = 1)

Measured on consumer x86_64 CPU hardware (Intel/AMD):

| Pipeline Stage | Implementation | Median Latency | P95 Latency | Memory / Operational Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Model Loading** | PyTorch vs ONNX | 14.60 ms | — | ONNX session initialization: 7.63 ms |
| **Pose Detection** | MediaPipe Tasks API | 23.60 ms | 29.89 ms | TFLite XNNPACK CPU delegate |
| **Feature Preprocessing** | Skeleton Normalization | 0.29 ms | 0.41 ms | Hip centering and torso scaling |
| **Classifier Inference** | **ONNX Runtime (CPU)** | **0.20 ms** | **0.28 ms** | **2.72× faster than PyTorch CPU (0.56 ms)** |
| **Uncertainty Policy** | Temperature + Rejection | 0.03 ms | 0.05 ms | Calibrated with fitted T = 1.627 |
| **Temporal Confirmation** | K-Window State Machine | 0.002 ms | 0.004 ms | Sliding confirmation buffer |
| **Grounded Explainer** | Template Rule Engine | 0.005 ms | 0.008 ms | Deterministic, zero GPU overhead |
| **End-to-End Processing** | Full Video Pipeline | — | — | **23.2 FPS (2.32× Real-Time at 10 Hz)** |
| **Peak RAM Footprint** | Complete Process | — | — | **561.2 MB resident set size** |


At 10 sampled frames/second, a window contains 30 frames and the stride is 5 frames. K=2 adds one stride (0.5 seconds) after the first positive window, before accounting for processing time. Physical fall-onset delay in the historical evaluation and confirmation policy delay in the UI are different measurements.
