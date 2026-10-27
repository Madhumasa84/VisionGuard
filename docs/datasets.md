# Dataset Provenance and Privacy Documentation

VisionGuard Lite incorporates three distinct dataset representations across its research and engineering pipeline.

---

## 1. GMDCSA-24 Dataset (Primary Real-Data Benchmark)

- **Dataset Name**: GMDCSA-24 v2.1: A Dataset for Human Fall Detection in Videos
- **Citation**:
  ```bibtex
  @article{alam2024gmdcsa24,
    title={GMDCSA24: A Dataset for Human Fall Detection in Videos},
    author={Alam, Ekram and Sufian, Abu and Dutta, Paramartha and Leo, Marco and Hameed, I. A.},
    journal={Data in Brief},
    volume={56},
    pages={110822},
    year={2024},
    publisher={Elsevier},
    doi={10.1016/j.dib.2024.110822}
  }
  ```
- **Zenodo DOI**: [10.5281/zenodo.13354453](https://doi.org/10.5281/zenodo.13354453)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Subjects**: 4 adult participants (`Subject 1`, `Subject 2`, `Subject 3`, `Subject 4`).
- **Total Videos**: 160 video recordings (80 Fall videos across forward, backward, lateral, stumble; 80 Activity of Daily Living [ADL] videos including sitting, walking, standing, bending).
- **Privacy & Redistribution Policy**:
  - Raw MP4 human video recordings total **2.0 GB** in disk size.
  - In compliance with open-source repository size constraints (< 100 MB per file) and responsible human data stewardship, raw video files are excluded via `.gitignore`.
  - Full extracted MediaPipe landmark archives (`experiments/gmdcsa24_real/data/extracted_poses/*.npz`, 8.9 MB) and 4-fold Leave-One-Subject-Out preprocessed window arrays (`experiments/gmdcsa24_real/data/processed_windows/`, 39 MB) are preserved in the repository for complete bit-for-bit experimental reproducibility.

---

## 2. SAFER-Activities Dataset (Gated External Benchmark)

- **Repository**: [`SAFER-Activities/SAFER-Activities`](https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities) (ECCV 2026)
- **Access Level**: Gated (requires manual approval from dataset maintainers).
- **License**: CC BY-NC-SA 4.0 (Non-Commercial, ShareAlike).
- **Extractor Schema**: YOLOv8x + ViTPose-H (COCO-17 landmark representation).
- **Integration**:
  - VisionGuard Lite includes [`scripts/download_real_safer.py`](../scripts/download_real_safer.py) and [`src/data/safer_real_pipeline.py`](../src/data/safer_real_pipeline.py) for automated ingestion once maintainer authorization is approved.

---

## 3. Synthetic Kinematics Demonstration Dataset

- **Directory**: `datasets/raw/coco_17/` and `datasets/processed/coco_17/`
- **Total Sequences**: 228 simulated sequences across 12 synthetic subjects (`P01` through `P12`).
- **Generation Method**: Programmatic kinematic trajectories with realistic joint angles, noise, and stumble profiles (`src/data/downloader.py`).
- **Purpose**: Autonomous end-to-end pipeline verification (sliding window extraction, ONNX export, temperature scaling, selective classification) without external network dependencies.
- **Scientific Status**: Identified strictly as synthetic demonstration sequences; never presented as clinical elderly fall data.
