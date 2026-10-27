# Model card: VisionGuard Lite TemporalCNN

The measurements below are historical research results. They do not evaluate the updated demo tracking-quality gate or establish clinical reliability. See [architecture](architecture.md) for current behavior and [benchmarks](benchmarks.md) for evaluation context.

## Model Details
- **Architecture**: Temporal Convolutional Network (1D TemporalCNN)
- **Input Channels**: 51 (17 COCO joints $\times$ 3 features: $[x, y, \text{visibility}]$)
- **Temporal Window Size**: 30 frames at 10 Hz (3.0-second sliding temporal window)
- **Parameters**: 66,274 parameters
- **Model Formats**:
  - PyTorch Checkpoint (`checkpoints/real_temporalcnn_best.pt`): 275.7 KB
  - Standalone ONNX FP32 (`exported_models/real_temporalcnn.onnx`): 258.9 KB
  - Standalone ONNX INT8 (`exported_models/real_temporalcnn_int8.onnx`): 71.4 KB
- **Post-Hoc Calibration**: Temperature Scaling with fitted $T = 1.627$ on validation logits.

---

## Intended Use
- **Primary Use Case**: Research demonstration and benchmarking of lightweight, CPU-first pose-based human activity recognition and fall detection on commodity laptops and edge devices.
- **Out-of-Scope Uses**:
  - Critical medical life-safety monitoring without caregiver oversight.
  - Emergency 911 dispatch automation.
  - Unsupervised elderly care in clinical or hospital environments.

---

## Training & Evaluation Data
- **Primary Benchmark Dataset**: GMDCSA-24 v2.1 (Alam et al., *Data in Brief* / PMC11416611, Zenodo: `10.5281/zenodo.13354453`).
  - Total Participants: 4 healthy adults (`Subject 1`, `Subject 2`, `Subject 3`, `Subject 4`).
  - Total Video Recordings: 160 videos (staged falls across directions: forward, backward, lateral, stumble; activities of daily living: walking, sitting, bending, lying).
  - Validation Protocol: 4-Fold Leave-One-Subject-Out (LOSO) cross-validation (zero subject leakage across train, val, and test).
- **Secondary Baseline Dataset**: Synthetic Kinematics Demonstration Dataset (12 synthetic participants, 228 sequences in `datasets/raw/coco_17/`).

---

## 4-Fold LOSO Benchmark Results on GMDCSA-24 (Real Human Data)

| Model Architecture | Parameters | Window F1 (Mean ± Std) | Window Recall | Window Precision | AUROC | Event-Level Recall | Unconfirmed FA/hr | Mean Detection Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TemporalCNN (Model B - Primary)** | **66,274** | **0.7570 ± 0.1175** | **0.8148** | **0.7149** | **0.8821** | **95.7%** | **119.72** | **1.66s** |
| **BaselineMLP (Model A)** | 100,322 | 0.7831 ± 0.1038 | 0.8966 | 0.6986 | 0.8926 | 96.7% | 140.43 | 1.49s |
| **SmallTCN (Model C)** | 67,154 | 0.6889 ± 0.0841 | 0.7513 | 0.6523 | 0.8516 | 89.9% | 131.83 | 1.61s |

*Fold 3 Highlight (Best Overall Test Model)*:
- Event Recall: **100.0% (21/21 falls detected)**
- Test Window F1: **0.9104** (AUROC: **0.9815**)
- Detection Delay: **1.81s**

### Calibration Analysis under Cross-Subject Shift
- **Pre-Scaling Mean ECE**: 63.3% across the 4 folds.
- **Post-Scaling Mean ECE ($T = 1.627$)**: 59.3%.
- **Finding**: While temperature scaling softens extreme logit outputs, cross-subject biometric variance across only 4 subjects leaves non-negligible calibration error (~59%). This empirical finding demonstrates why raw probability thresholding cannot guarantee operational safety, motivating further research into confirmation and rejection policies; neither establishes operational safety.

---

## Temporal Confirmation Policy Ablation

Continuous evaluation over simulated continuous activity monitoring streams:

| Confirmation Policy | Window Criteria | Event Recall | Missed Falls | False Alarm Rate | Mean Physical Detection Delay | Operational Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw Unfiltered ($K=1$)** | $\ge 1$ window | 96.1% (74/77) | 3 | 119.72 FA/hr (~1 / 30.1s) | 1.66s | Unusable without confirmation |
| **Confirmed Filter ($K=2$)** | $\ge 2$ windows | **92.2% (71/77)** | 6 | **95.00 FA/hr (~1 / 37.9s)** | **2.18s** | **Demo default; high false-alarm rate remains** |
| **Strict Filter ($K=3$)** | $\ge 3$ windows | 87.0% (67/77) | 10 | 87.53 FA/hr (~1 / 41.1s) | 2.67s | Conservative setting |
| **High Precision ($K=4$)** | $\ge 4$ windows | 70.1% (54/77) | 23 | 83.80 FA/hr (~1 / 43.0s) | 3.14s | Elevated false-negative risk |

---

## Explanation Engine Architecture

1. **Deterministic Rule Engine (`TemplateExplainer` - Default Runtime)**:
   - Previously measured execution: **0.005 ms** median latency on CPU.
   - Summarizes activity classification, confidence, visibility, uncertainty, and confirmation evidence. The demo does not supply measured joint angles or torso velocity to this engine.
   - Checked by `EvidenceConsistencyChecker` for uncertainty wording and selected unsupported claims; the result is reported in diagnostics.
2. **Optional Small Language Model (`QwenSLMExplainer` - Research Mode)**:
   - Optional text-generation prototype using Qwen2.5-0.5B-Instruct through Transformers. The current implementation does not implement quantization or a llama.cpp backend.
   - Not selectable in the demo UI. Loading the optional model may require network access and additional dependencies.

---

## Measured Consumer CPU Latencies (Batch Size = 1)

Benchmarked on Intel/AMD consumer x86_64 CPU:
- **Pose Extraction (MediaPipe Tasks)**: 23.60 ms median / 29.89 ms P95
- **Feature Preprocessing**: 0.29 ms median
- **TemporalCNN ONNX Runtime Inference**: **0.20 ms median** / 0.28 ms P95
- **TemporalCNN PyTorch CPU Inference**: 0.56 ms median / 0.84 ms P95
- **Uncertainty Estimation**: 0.03 ms median
- **Temporal Confirmation Filtering**: 0.002 ms median
- **Template Explanation Engine**: 0.005 ms median
- **End-to-End Pipeline Throughput**: **23.2 FPS** (2.32× real-time at 10 Hz)
- **Peak RAM Footprint**: 561.2 MB
