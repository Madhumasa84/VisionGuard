# 📊 SAFER-Activities Real Dataset Audit & Pipeline Integration Report

**Report Date**: October 2026  
**Target Repository**: [`SAFER-Activities/SAFER-Activities`](https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities) (ECCV 2026)  
**Authenticated User**: `Masa84` (`smkp84@gmail.com`)  
**Project Path**: `/content/VisionGuardLite/`

---

## 1. Step 1 — Official Dataset Access & Status

### 1.1 Authentication & Gated Status
* We successfully authenticated with Hugging Face Hub using the provided user credentials.
* **Access Diagnostic**:
  * Repository ID: `SAFER-Activities/SAFER-Activities`
  * Gating Mode: **`gated: manual`** (Requires explicit approval by repository maintainers)
  * Current Authorization Status: **Blocked with `403 Client Error (Forbidden)`**

```text
huggingface_hub.errors.GatedRepoError: 403 Client Error.
Access to dataset SAFER-Activities/SAFER-Activities is restricted and you are not in the authorized list.
Visit https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities to ask for access.
```

### 1.2 Exact User Action Required for Approval
The repository requires manual request approval. To unlock access:
1. Navigate to: [https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities](https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities)
2. Fill in the required gated fields:
   * **Organization / Institution** (e.g. University / Research Lab / Independent)
   * **Intended Use** (e.g. Academic research on edge fall detection and activity recognition)
3. Accept the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)** license.
4. Once the dataset maintainers approve your request, the stored token will automatically be authorized to download all files.

---

## 2. Step 2 — Repository Structure & Size Inspection

Through the Hugging Face API, we successfully inspected the remote repository structure and file manifests:

```text
SAFER-Activities/SAFER-Activities
├── splits/                                    # Official train/test subject & view splits
│   ├── non_wheelchair_subject_train.txt      # 0.01 MB
│   ├── non_wheelchair_subject_test.txt       # 0.00 MB
│   ├── non_wheelchair_view_train.txt         # 0.01 MB
│   └── non_wheelchair_view_test.txt          # 0.00 MB
├── pose_bboxes/
│   └── 3d_keypoints_pickle_ntu_format.zip    # 2,899.36 MB (~2.9 GB, 2D + lifted 3D poses)
├── raw/
│   └── Safer-Activities-Full-Dataset/
│       └── normal/
│           └── CSVs/                         # 60 annotation files (e.g. apr_25_2023_fall_p005.csv)
├── extracted_features/                       # Frozen per-video RGB features
└── wheelchair_keypoints/                     # Dedicated wheelchair subset
```

### 2.1 Pose & Keypoint Representation Details
* **Pose Extraction Pipeline**: 2D poses were pre-extracted using **YOLOv8x (detection) + ViTPose-H (pose estimation)**.
* **Pose Format**: MMAction2 / PYSKL pickle format (`keypoint` of shape $T \times 17 \times 2$, `keypoint_score` of shape $T \times 17$, and `bboxes`).
* **Annotation Format**: CSVs containing `Action`, `Start Time`, `End Time` with sub-second timestamps (`HH:MM:SS.mmm`).

---

## 3. Step 3 — Real-Data Processing Pipeline Implementation

We created [`src/data/safer_real_pipeline.py`](file:///content/VisionGuardLite/src/data/safer_real_pipeline.py), specifically tailored for the SAFER-Activities schema:

1. **Sub-second Timestamp Alignment**:
   $$\text{Start Frame} = \text{round}\left((3600 \cdot H + 60 \cdot M + S + \frac{mmm}{1000}) \times \text{FPS}\right)$$
2. **Window Construction & Fall Labeling**:
   - 30-frame temporal sliding windows (3.0s at 10 FPS).
   - If window overlap with annotated fall event $\ge 50\% \implies \text{Label} = 1$ (Fall).
   - If window overlap is $0\% \implies \text{Label} = 0$ (Non-fall).
   - Ambiguous boundary windows ($0 < \text{overlap} < 50\%$) are excluded to prevent label noise.
3. **Event-Level Metric Aggregation**:
   - Overlapping positive windows belonging to the same fall event are grouped into a single true positive event to prevent false metric inflation.
   - Measures mean detection delay in seconds from the real physical fall onset.

---

## 4. Step 5 — Cross-Extractor Pose Consistency Audit

We created [`src/data/pose_schema_audit.py`](file:///content/VisionGuardLite/src/data/pose_schema_audit.py) to audit the mapping between the real data pose extractor (ViTPose-H COCO-17) and the deployment inference extractor (MediaPipe-33):

### 4.1 Landmark Alignment Table

| COCO-17 Index | Joint Name | MediaPipe-33 Source Index | Verification |
| :---: | :--- | :---: | :---: |
| 0 | Nose | 0 | Exact match |
| 1, 2 | Left / Right Eye | 2, 5 | Eye centers |
| 3, 4 | Left / Right Ear | 7, 8 | Ear positions |
| 5, 6 | Left / Right Shoulder | 11, 12 | Acromion joints |
| 7, 8 | Left / Right Elbow | 13, 14 | Elbow joints |
| 9, 10 | Left / Right Wrist | 15, 16 | Wrist centers |
| 11, 12 | Left / Right Hip | 23, 24 | Greater trochanter / hip centers |
| 13, 14 | Left / Right Knee | 25, 26 | Knee joints |
| 15, 16 | Left / Right Ankle | 27, 28 | Lateral malleolus / ankle joints |

### 4.2 Automated Testing
Added unit tests in [`tests/test_real_pipeline.py`](file:///content/VisionGuardLite/tests/test_real_pipeline.py), validating timestamp parsing, index mapping, and event-level recall. Total passing tests: **17/17**.

---

## 5. Summary of Deliverables & Ready-to-Run Scripts

1. **Authentication Utility**: [`download_real_safer.py`](file:///content/VisionGuardLite/download_real_safer.py)
   ```bash
   python3 /content/VisionGuardLite/download_real_safer.py
   ```
2. **Real Data Pipeline**: [`src/data/safer_real_pipeline.py`](file:///content/VisionGuardLite/src/data/safer_real_pipeline.py)
3. **Cross-Extractor Auditor**: [`src/data/pose_schema_audit.py`](file:///content/VisionGuardLite/src/data/pose_schema_audit.py)
4. **Test Suite**: [`tests/test_real_pipeline.py`](file:///content/VisionGuardLite/tests/test_real_pipeline.py) (**17/17 passing**)
5. **Preserved Checkpoints**: Existing synthetic demonstration checkpoints and metrics remain untouched in `checkpoints/` and `results/metrics/`. Real-data models will be saved in `checkpoints/real/` and `results/metrics/real/`.
