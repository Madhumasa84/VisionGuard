# VisionGuard Lite: Repository Audit Report (Phase 1)

**Audit Date**: October 2026  
**Auditor**: Antigravity Autonomous Verification Suite  
**Scope**: Codebase structure, dependencies, checkpoints, pose representations, data privacy, and platform portability.

---

## 1. Executive Summary

VisionGuard Lite contains two distinct research phases:
1. **Synthetic Kinematics Demonstration Pipeline**: Validated on 12 synthetic subjects (`P01`–`P12`, 228 sequences) in `datasets/raw/coco_17/`, demonstrating baseline model architectures (`BaselineMLP`, `TemporalCNN`, `SmallTCN`), probability calibration (Temperature Scaling), selective prediction, and ONNX runtime export.
2. **Real GMDCSA-24 LOSO Benchmark**: Implemented on real staged fall and activity-of-daily-living (ADL) video recordings across 4 participants (`Subject 1`–`Subject 4`, 160 videos) under a Leave-One-Subject-Out (LOSO) cross-validation protocol.

Before releasing to public GitHub as a reliable, CPU-first application, the repository requires comprehensive portability remediation, clean decoupling of synthetic vs real artifacts, and elimination of Colab-specific assumptions.

---

## 2. Identified Repository Deficiencies

### 2.1 Colab-Specific Imports & Hardcoded Paths
* **Hardcoded `/content/drive/MyDrive/VisionGuardLite` and `/content/VisionGuardLite`**:
  * `demo/app.py`: Hardcoded `sys.path.insert(0, '/content/drive/MyDrive/VisionGuardLite')`, `PROJECT_DIR`, and checkpoint paths.
  * `src/inference/mediapipe_extractor.py`: Default `model_path` points to `/content/drive/MyDrive/...`.
  * `src/training/run_experiments.py`: Hardcoded Colab project and dataset paths.
  * `src/data/downloader.py` & `download_real_safer.py`: Hardcoded destination `/content/VisionGuardLite/...`.
  * `tests/test_slm_and_onnx.py`: Hardcoded ONNX model path to `/content/drive/...`.
  * `experiments/gmdcsa24_real/src/*.py`: Scripts hardcode `/content/VisionGuardLite/...`.
  * `experiments/gmdcsa24_real/data/extracted_poses/manifest.json`: All 160 file records point to `/content/VisionGuardLite/...`.
  * `datasets/raw/coco_17/manifest.json`: All 228 file records point to `/content/drive/...`.
  * `README.md`: Execution instructions specify Colab paths (`PYTHONPATH=/content/drive/...`).

### 2.2 Linux-Only Dependencies & Windows Compatibility
* `requirements.txt` does not distinguish CPU-only builds. Unconstrained `torch` installation attempts to pull multi-gigabyte CUDA wheels or fail on systems without GPU toolchains.
* Video encoding in OpenCV uses `*'mp4v'`, which can encounter codec failures on Windows without fallback handling (e.g., portable FourCC negotiation or image-sequence fallback).
* Lack of Windows PowerShell automation scripts (`setup_windows.ps1`, `run_local.ps1`, `run_tests.ps1`).

### 2.3 Broken Imports & Path Resolution
* Scripts rely on manual `sys.path.insert()` rather than package root discovery (`pathlib.Path(__file__).resolve().parents[...]`).
* If `demo/app.py` is launched from outside the root directory, internal imports fail unless installed as a package or resolved via `pathlib`.

### 2.4 Duplicate Files & Unnecessary Generated Artifacts
* **734 NTFS Zone Identifier files (`*:Zone.Identifier`)** present across video and code directories, created by Windows NTFS alternate data streams.
* Duplicate notebook files: `./VisionGuard_Lite_Colab.ipynb` and `./notebooks/VisionGuard_Lite_Colab.ipynb`.
* Duplicate exported ONNX models in `exported_models/` (`temporal_cnn.onnx`, `temporal_cnn_clean.onnx`, `temporal_cnn_embedded.onnx`, etc.).
* Python bytecode artifacts (`__pycache__`) compiled under Python 3.13 checked into repository.
* Temporary demo outputs (`demo/temp_timeline.png`, `demo/annotated_output.mp4`) tracked in git.

### 2.5 Dataset Redistribution & Privacy Risks
* `experiments/gmdcsa24_real/data/raw/gmdcsa24` contains **2.0 GB** of raw participant video recordings (`Subject 1` through `Subject 4`, 143+ MP4 files).
* Committing 2 GB of human subject videos violates GitHub repository and file size limits (100 MB max per file, 1–2 GB recommended repo limit) and introduces privacy/consent redistribution risks for raw identifiable human video.
* Raw video data must be excluded via `.gitignore`, while preserving extracted landmark arrays (`.npz`/`.npy`) and automated download scripts.

### 2.6 Credential & Personal Information Audit
* No exposed secret API keys or private tokens were discovered.
* `REAL_DATA_REPORT.md` references the user's personal email address (`smkp84@gmail.com`) and username (`Masa84`). This should be sanitized for open-source publication.
* A `.env.example` template is required for configuring optional Hugging Face tokens and environment variables.

### 2.7 Model Checkpoint Alignment
* `demo/app.py` previously loaded `checkpoints/temporal_cnn_best.pt`, which was trained on synthetic data.
* Phase 3 requires the application to load the real GMDCSA-trained model weights, with clear separation between MediaPipe-33 and COCO-17 coordinate frames.

---

## 3. Action Plan & Next Phases

1. **Phase 2**: Implement CPU-only configuration (`requirements-local.txt`, `requirements-dev.txt`, `.env.example`, and PowerShell setup/run scripts).
2. **Phase 3**: Refactor `demo/app.py` and `src/` to use portable `pathlib.Path`, load the real GMDCSA-trained model, handle temporal confirmation, and run locally at `http://127.0.0.1:7860`.
3. **Phase 4**: CPU-first latency benchmarking (batch size 1, real video input, ONNX Runtime vs PyTorch parity).
4. **Phase 5**: UI polish with telemetry, timeline visualization, and grounded explanations.
5. **Phase 6**: Scientific integrity disclosures (false alarm rate: 132.4 FA/hr = 1 every 27.2 seconds; 4 subjects; lack of clinical validation).
6. **Phase 7 & 8**: Engineering cleanup (`.gitignore`, removal of Zone.Identifier and cache files, test suites, documentation, and `LOCAL_VALIDATION_REPORT.md`).
