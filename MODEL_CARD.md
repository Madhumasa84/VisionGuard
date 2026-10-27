# Model Card: VisionGuard Lite TemporalCNN

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
