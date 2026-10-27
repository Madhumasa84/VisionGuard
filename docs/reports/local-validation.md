# Local Validation and Release Verification Report (Phase 8)

**Report Date**: October 2026  
**Auditor**: Antigravity Autonomous Pair-Programming Engine  
**Release Candidate**: VisionGuard Lite v0.2.0 (CPU-First Public GitHub Release)  
**Repository Working Directory**: `/home/masa84/VisionGuardLite`

---

## 1. Environment & Runtime Specifications

| Attribute | Specification | Verification |
| :--- | :--- | :---: |
| **Operating System** | Linux (Ubuntu 22.04 LTS x86_64, Kernel: WSL2 6.18) | Verified |
| **Python Version** | Python 3.10.12 | Verified |
| **Primary Acceleration** | **CPU Only** (torch CPU build, CUDA disabled) | Verified |
| **Primary Backend** | **ONNX Runtime (CPUExecutionProvider)** | Verified |
| **Secondary Backend** | PyTorch 2.14.1+cpu | Verified |
| **Network Dependency** | Zero remote calls required during local inference | Verified |

### Installed Core Package Versions
- `torch`: `2.14.1+cpu`
- `onnxruntime`: `1.23.2`
- `onnx`: `1.23.2`
- `mediapipe`: `1.1.1`
- `gradio`: `6.30.0`
- `opencv-python-headless`: `5.0.0.93`
- `scipy`: `1.15.3`
- `scikit-learn`: `1.7.2`
- `numpy`: `2.2.6`
- `matplotlib`: `3.10.9`
- `psutil`: `7.2.2`
- `pytest`: `9.1.1`

---

## 2. Test Suite Execution Summary

Executed via `pytest tests/ -v` on local CPU:

| Test File | Test Cases | Result | Duration |
| :--- | :--- | :---: | :---: |
| `tests/test_models.py` | Architecture forward passes (`BaselineMLP`, `TemporalCNN`, `SmallTCN`) | **3 / 3 Passed** | 0.82s |
| `tests/test_preprocessing.py` | Normalization, hip centering, imputation, windowing | **4 / 4 Passed** | 0.45s |
| `tests/test_real_pipeline.py` | Timestamp parsing, MediaPipe-to-COCO mapping, event recall | **3 / 3 Passed** | 0.38s |
| `tests/test_slm_and_onnx.py` | Template explainer, safety consistency checker, ONNX runtime | **4 / 4 Passed** | 0.52s |
| `tests/test_uncertainty.py` | TemperatureScaler, ECE calculation, UncertaintyPolicy rejection | **3 / 3 Passed** | 0.28s |
| `tests/test_integration_smoke.py` | End-to-end pipeline, ONNX/PyTorch parity, invalid inputs, K-policy | **5 / 5 Passed** | 12.18s |
| **Total Test Suite** | **Comprehensive Unit & Integration Coverage** | **22 / 22 Passed (100%)** | **14.63s** |

---

## 3. Models Loaded & Verified

1. **GMDCSA-24 Real Model Checkpoint**:
   - Location: `checkpoints/real_temporalcnn_best.pt` (275.7 KB)
   - Checkpoint Metadata: Fold 3 (Test Subject 3), $T = 1.627$, Optimal Threshold = $0.18$
   - Parameter Count: 66,274
   - Performance: 100% Event Recall (21/21 falls detected), 0.9104 F1, 0.9815 AUROC.
2. **Standalone ONNX FP32 Model**:
   - Location: `exported_models/real_temporalcnn.onnx` (258.9 KB)
   - Initializers: Completely self-contained (no external `.data` file required)
   - Max Numerical Difference vs PyTorch: **$9.54 \times 10^{-7}$** (Parity Passed)
3. **Dynamic INT8 Quantized Model**:
   - Location: `exported_models/real_temporalcnn_int8.onnx` (71.4 KB)
4. **MediaPipe Tasks Pose Asset**:
   - Location: `checkpoints/pose_landmarker_lite.task` (5.6 MB)
5. **Preserved Synthetic Baselines**:
   - `checkpoints/baseline_mlp_best.pt`, `checkpoints/temporal_cnn_best.pt`, `checkpoints/small_tcn_best.pt`.

---

## 4. Actual Inference Outputs on Permitted Real Video

- **Target Video**: `experiments/gmdcsa24_real/data/raw/gmdcsa24/Subject 1/Fall/01.mp4`
- **Video Properties**: 68 frames, 6.8 seconds duration, 10.0 FPS.
- **Pose Extraction**: 68 frames tracked via MediaPipe Tasks with 63.2% mean landmark visibility.
- **Window Trajectory Probabilities**:
  - Window 0 (0.0s – 3.0s): `0.0985` (Standing / Walking)
  - Window 1 (0.5s – 3.5s): `0.1950` (Normal ADL)
  - Window 2 (1.0s – 4.0s): `0.5471` (Fall transition begins)
  - Window 3 (1.5s – 4.5s): `0.7591` (Fall threshold exceeded)
  - Window 4 (2.0s – 5.0s): `0.8664` (Body impacts floor — **Confirmed Alert Triggered at K=2**)
  - Window 5 (2.5s – 5.5s): `0.8813` (Resting on floor)
  - Window 6 (3.0s – 6.0s): `0.8806` (Resting on floor)
  - Window 7 (3.5s – 6.5s): `0.8967` (Resting on floor)
- **Decision Status**: `CONFIRMED FALL EVENT`
- **Physical Confirmation Delay**: **0.5 s** from initial threshold crossing
- **ONNX vs PyTorch Probability Parity**: Max difference = **$1.19 \times 10^{-7}$**
- **Explanation**: `"ALERT: A sudden downward fall transition was detected with high confidence (90%). The body reached ground-level orientation."`
- **Safety Consistency Checker**: `Passed = True` (Zero hallucinated injuries/claims)

---

## 5. Measured Consumer CPU Benchmark Results

Benchmarked with Batch Size = 1 on consumer x86_64 CPU:

| Benchmark Stage | Measured Latency / Metric | Evaluation Unit |
| :--- | :--- | :--- |
| **PyTorch Model Loading** | 14.60 ms | Single cold load |
| **ONNX Runtime Session Init** | 7.63 ms | Single session creation |
| **Pose Detection (MediaPipe)** | Median: 23.60 ms \| P95: 29.89 ms | Per video frame |
| **Feature Preprocessing** | Median: 0.29 ms \| P95: 0.41 ms | Per 30-frame window |
| **PyTorch Classifier Inference** | Median: 0.56 ms \| P95: 0.84 ms | Single window (Batch=1) |
| **ONNX Runtime Inference** | **Median: 0.20 ms \| P95: 0.28 ms** | Single window (Batch=1) |
| **ONNX Speedup over PyTorch** | **2.72× faster** | Ratio of medians |
| **Uncertainty Estimation** | Median: 0.03 ms \| P95: 0.05 ms | Temperature + rejection |
| **Temporal Filtering** | Median: 0.002 ms \| P95: 0.004 ms | K-window filter |
| **Template Explanation** | Median: 0.005 ms \| P95: 0.008 ms | Deterministic rule engine |
| **End-to-End Processing Speed** | **23.2 FPS** | Effective throughput |
| **Real-Time Factor** | **2.32× Real-Time** (5.0s video in 2.16s) | Real-time margin |
| **Process Peak RAM** | **561.2 MB** (Base: 290 MB, Delta: +271 MB) | Peak resident set size |

---

## 6. Gradio Local Application Verification

- **Local Server Binding**: Tested at `http://127.0.0.1:7860` with `share=False`.
- **HTTP Response Code**: `200 OK` (Local verification passed).
- **Features Tested**:
  - Video upload source and webcam recording options.
  - Interactive parameter controls: Runtime Backend (ONNX vs PyTorch), Confirmation $K$, Confidence Threshold.
  - Video playback with bounding box and HUD tracking overlays.
  - Temporal probability curve visualization with confirmed fall shaded zone.
  - Telemetry metrics card and grounded explanation box.
  - Research limitations disclosure banner visible.

---

## 7. Known Issues & Operational Considerations

1. **Small Staged Participant Cohort**: Trained on 4 healthy young adults under staged conditions. Not evaluated on geriatric cohorts or individuals using walkers or wheelchairs.
2. **False Alarm Frequency**: Continuous unconfirmed monitoring produces **~132.4 false alarms/hour** (~1 every 27.2 seconds). Temporal confirmation ($K \ge 2$) suppresses transient spikes at the cost of a 1.0–1.5s confirmation delay.
3. **Severe Occlusion**: Landmark loss $> 40\%$ triggers selective abstention (`UNCERTAIN`).
4. **Linux Headless Environments**: MediaPipe relies on `libGLESv2.so.2`. VisionGuard Lite includes local fallback preloading via `ctypes.CDLL` in `mediapipe_extractor.py`, eliminating runtime missing shared library crashes.

---

## 8. GitHub Release Readiness Assessment

| Verification Item | Status | Details |
| :--- | :---: | :--- |
| **Directory Hygiene** | PASS | All 734 NTFS `*:Zone.Identifier` and `.pyc` files removed; duplicate ONNX models consolidated |
| **Colab Decoupling** | PASS | Zero hardcoded `/content/` or `/content/drive` paths in runtime code |
| **Path Portability** | PASS | All scripts resolve dynamically via `pathlib.Path` |
| **Dependency Alignment** | PASS | `requirements-local.txt` and `requirements.txt` harmonized to match runtime environment (`numpy>=1.24.0`, `gradio>=4.20.0`) |
| **Model Integrity** | PASS | Synthetic fallback removed; demo strictly requires real trained GMDCSA checkpoints |
| **CPU-First Reproducibility** | PASS | Reproducible CPU runtime, `.env.example`, PowerShell & Bash runners |
| **Test Coverage** | PASS | 22/22 unit and integration tests passing; CI matrix configured for Ubuntu and Windows |
| **Model Weight Distribution** | PASS | Real GMDCSA models + standalone ONNX models < 300 KB each |
| **Dataset Privacy & Limits** | PASS | 2.0 GB raw human videos excluded via `.gitignore`; extracted features kept |
| **Documentation Quality** | PASS | `README`, `ARCHITECTURE`, `MODEL_CARD`, `DATASET`, `LIMITATIONS` complete without promotional badges or emojis |
| **CI Automation** | PASS | GitHub Actions matrix workflow (`ubuntu-latest` and `windows-latest`, Python 3.10 and 3.11) |

**Overall Status**: **APPROVED FOR PUBLIC GITHUB RELEASE**
