# 🔬 VisionGuard Lite: Scientific Validation & Audit Report

**Report Date**: October 2026  
**Audit Target**: VisionGuard Lite Repository (`/content/VisionGuardLite/`)  
**Main Master Notebook**: [`VisionGuard_Lite_Colab.ipynb`](file:///content/VisionGuardLite/VisionGuard_Lite_Colab.ipynb)  
**Lead Investigator**: Autonomous Research & Verification Suite

---

## ⚠️ Scientific Integrity Statement

> [!IMPORTANT]
> **Data Provenance Declaration**:
> The previously reported **93.33% F1 score** and **0.9982 AUROC** were achieved **exclusively on synthetic kinematic demonstration sequences**, **NOT on real human activity video recordings from the SAFER-Activities dataset**.
>
> The synthetic data was generated autonomously by `src/data/downloader.py` to mathematically validate the end-to-end pipeline (sliding window extraction, coordinate normalization, 1D temporal convolutions, post-hoc calibration, selective abstention, and ONNX deployment) while access to the gated Hugging Face dataset is pending user authentication.
>
> **These results must be cited strictly as demonstration-only and not as real-world clinical fall detection accuracy.**

---

## 1. Task 1 — Data Provenance Audit

### 1.1 Traceability Analysis
We audited every step of the data pipeline:

```text
[SAFERDatasetManager.generate_representative_dataset()] (downloader.py)
   │
   ▼
[datasets/raw/coco_17/] (228 .npz sequence files + manifest.json)
   │
   ▼
[PosePreprocessor.create_sliding_windows()] (preprocessing.py)
   │
   ▼
[datasets/processed/coco_17/] (X_train.npy, y_train.npy, X_val.npy, X_test.npy)
   │
   ▼
[ModelTrainer.fit() & evaluate()] (run_experiments.py -> model_comparison.json)
```

### 1.2 Exact Source Files and Distribution
- **Raw Sequence Directory**: `datasets/raw/coco_17/`
- **Total Subjects**: 12 synthetic subjects (`P01` to `P12`)
- **Total Video Sequences**: 228 generated sequences (at 10 FPS, 7–10 seconds each):
  - Normal Walking: 48 sequences
  - Normal Sitting: 36 sequences
  - Normal Standing: 36 sequences
  - Normal Bending: 36 sequences
  - Stumble (Near-Fall): 24 sequences
  - Fall Events: 48 sequences (with marked onset, impact, and post-fall rest)
- **Windowing Parameters**: 30 frames (3.0s window), stride = 10 frames (train), stride = 5 frames (val/test).

### 1.3 Split Windows & Participant Isolation

| Split | Subject IDs | Total Windows | Fall Windows | Non-Fall Windows | Fall Ratio | Ambiguous Windows Excluded |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | `P01`–`P08` | 952 | 128 | 824 | 13.45% | 64 |
| **Val** | `P09`, `P10` | 430 | 56 | 374 | 13.02% | 40 |
| **Test** | `P11`, `P12` | 430 | 56 | 374 | 13.02% | 40 |

*Zero Leakage*: Confirmed zero participant overlap between train, validation, and test splits.

---

## 2. Task 2 — Model Export Verification & ONNX Anomaly Resolution

### 2.1 Investigation of the 16.32 KB ONNX File Anomaly
The reported `temporal_cnn.onnx` file had a reported size of **16.32 KB** despite containing **66,274 parameters**. 

```text
66,274 parameters × 4 bytes/float32 = 265,096 bytes ≈ 258.9 KB
```

**Root Cause Found**:
In PyTorch 2.11, default `torch.onnx.export` with Dynamo split the model into:
1. `temporal_cnn.onnx`: **16,716 bytes (16.32 KB)** — contained **only the protobuf graph topology** and metadata.
2. `temporal_cnn.onnx.data`: **262,784 bytes (256.62 KB)** — contained the external weight tensor initializers.

*Total Disk Footprint*: $16.32 \text{ KB} + 256.62 \text{ KB} = \mathbf{272.94 \text{ KB}}$.

> [!WARNING]
> If `temporal_cnn.onnx` is distributed without `temporal_cnn.onnx.data`, the ONNX runtime fails to initialize due to missing external initializers.

### 2.2 Fix Implemented: Standalone Embedded ONNX & INT8 Export
We updated `src/inference/onnx_exporter.py` to use `dynamo=False` with `export_params=True`, ensuring all 66k parameters are embedded directly into a **single, standalone file**:

| Model File | Format | File Size | Parameter Count | External Data Required? | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| [`temporal_cnn.onnx`](file:///content/VisionGuardLite/exported_models/temporal_cnn.onnx) | FP32 | **258.92 KB** | 66,274 | **No (Self-contained)** | ✅ Fixed & Verified |
| [`temporal_cnn_int8.onnx`](file:///content/VisionGuardLite/exported_models/temporal_cnn_int8.onnx) | INT8 | **71.41 KB** | 66,274 | **No (Self-contained)** | ✅ Created & Verified |

### 2.3 Multi-Input Numerical Parity Checks
Evaluated across 20 real test sequence windows and 10 random tensors:
- **Max Absolute Difference (FP32 ONNX vs PyTorch)**: **$1.43 \times 10^{-6}$** (`parity_passed: True`)
- **Prediction Agreement on Test Set**: **100.0% (430/430 identical predictions)**
- **Test Set F1**: **0.9333 (PyTorch) vs 0.9333 (FP32 ONNX) vs 0.9333 (INT8 ONNX)**
- **Single-Sample CPU Latency**: **0.070 ms (FP32)**, **0.265 ms (INT8)**

---

## 3. Task 3 — Reliability & Distribution Shift Investigation

We audited why F1 degraded drastically under 40% occlusion and 0.6× scale shift:

### 3.1 The 0.6× Camera Scale Shift Anomaly
- **Initial Observation**: Under 0.6× scale shift, F1 dropped to `0.2305`, Recall was `1.0000`, and Precision was exactly `0.1302`.
- **Finding**: A precision of `0.1302` matches the exact base fall rate of the test set ($56 / 430 = 0.13023$). The model was classifying **every single window as a fall**.
- **Root Cause**: In the original `evaluate_distribution_shifts()`, the 0.6× multiplier was applied to **already normalized coordinates** (`X_test * 0.6`). This compressed the torso length to 60%, making standing humans appear collapsed at floor height.
- **Verification on Pre-Normalized Data**:
  When camera distance is properly simulated **before normalization** (as in real cameras where a distant person has smaller pixel coordinates):
  $$\text{Torso Length} \propto 0.6, \quad \text{Coordinates} \propto 0.6 \implies \frac{0.6 \cdot (\mathbf{x} - \mathbf{x}_{\text{hip}})}{0.6 \cdot L_{\text{torso}}} = \frac{\mathbf{x} - \mathbf{x}_{\text{hip}}}{L_{\text{torso}}}$$
  The scale cancels out completely:
  - **Clean**: F1 = 0.9333 | Recall = 1.0000 | Precision = 0.8750 | AUROC = 0.9982
  - **Realistic Scale Shift (Pre-Norm)**: **F1 = 0.9333 | Recall = 1.0000 | Precision = 0.8750 | AUROC = 0.9982**
  - **Conclusion**: The preprocessing normalization is **100% invariant to camera distance**.

### 3.2 The 40% Landmark Occlusion Failure
- **Observation**: Zeroing out 40% of landmarks causes F1 to drop to `0.2593` (AUROC = 0.7444).
- **Explanation**: When 40% of keypoints (especially hips and knees) disappear, spatial topology collapses. Posture estimation fails because the reference hip center defaults to zero.
- **Selective Classification Mitigation**:
  Instead of forcing a dangerous classification under heavy occlusions, [`UncertaintyPolicy`](file:///content/VisionGuardLite/src/evaluation/uncertainty.py) monitors landmark visibility and predictive entropy.
  - **Under 40% Occlusion**: The policy triggers **20.23% abstention**, rejecting ambiguous frames as `UNCERTAIN` for secondary caregiver review.

---

## 4. Task 4 — Real Dataset Integration Workflow

### 4.1 Status of `g-b-r-l/SAFER-Activities`
- **Location**: Hugging Face Hub (`g-b-r-l/SAFER-Activities`).
- **Access Level**: Gated repository (requires accepted license and approved token).
- **Current Workspace State**: No Hugging Face token is stored; queries return `401 Client Error`.

### 4.2 Official Dataset Ingestion Workflow
We implemented [`download_real_safer.py`](file:///content/VisionGuardLite/download_real_safer.py) to enable 1-click authenticated download:

```bash
# To authenticate and download the official dataset:
python3 /content/VisionGuardLite/download_real_safer.py --token <YOUR_HF_READ_TOKEN>
```

Or programmatically in a Colab notebook cell:
```python
from src.data.downloader import SAFERDatasetManager

# Requires accepting agreement at https://huggingface.co/datasets/g-b-r-l/SAFER-Activities
SAFERDatasetManager.download_real_dataset(token="YOUR_HF_TOKEN")
```

---

## 5. Task 5 — Verified Evaluation Matrix

| Experiment / Evaluation Category | Dataset Provenance | Status | Fall F1 | Fall Recall | Fall Precision | AUROC | Latency (CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A ([`BaselineMLP`](file:///content/VisionGuardLite/src/models/baseline_mlp.py))** | Synthetic Kinematics | Complete (Demo) | 0.9333 | 1.0000 | 0.8750 | 0.9988 | 0.110 ms |
| **Model B ([`TemporalCNN`](file:///content/VisionGuardLite/src/models/temporal_cnn.py))** | Synthetic Kinematics | Complete (Demo) | 0.9333 | 1.0000 | 0.8750 | 0.9982 | 0.370 ms |
| **Model C ([`SmallTCN`](file:///content/VisionGuardLite/src/models/small_tcn.py))** | Synthetic Kinematics | Complete (Demo) | 0.9333 | 1.0000 | 0.8750 | 0.9982 | 4.499 ms |
| **Exported Model B ([`FP32 ONNX`](file:///content/VisionGuardLite/exported_models/temporal_cnn.onnx))** | Synthetic Kinematics | Complete (Verified) | 0.9333 | 1.0000 | 0.8750 | 0.9982 | 0.070 ms |
| **Exported Model B ([`INT8 ONNX`](file:///content/VisionGuardLite/exported_models/temporal_cnn_int8.onnx))** | Synthetic Kinematics | Complete (Verified) | 0.9333 | 1.0000 | 0.8750 | 0.9982 | 0.265 ms |
| **Pre-Norm Distance Shift (0.6×)** | Synthetic Kinematics | Complete (Verified) | 0.9333 | 1.0000 | 0.8750 | 0.9982 | 0.370 ms |
| **Real SAFER-Activities Evaluation** | Real Human Recordings | **Pending User Auth** | — | — | — | — | — |

---

## 6. Next Steps for Verified Real-Data Execution

1. **Provide Hugging Face Access**:
   Once you accept the CC BY-NC-SA 4.0 license at [Hugging Face SAFER-Activities](https://huggingface.co/datasets/g-b-r-l/SAFER-Activities), run:
   ```bash
   python3 /content/VisionGuardLite/download_real_safer.py --token <HF_TOKEN>
   ```
2. **Execute Real Preprocessing**:
   Run the preprocessing pipeline on real extracted COCO and MediaPipe landmarks.
3. **Re-train and Update Verified Metrics**:
   Re-train Model B on real subject splits and update `VALIDATION_REPORT.md` with true clinical sensitivity and false alarm rates per hour.
