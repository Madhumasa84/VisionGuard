"""
Script to generate VisionGuard_Lite_Colab.ipynb cleanly using nbformat
"""

import os
import nbformat as nbf

PROJECT_DIR = '/content/drive/MyDrive/VisionGuardLite'

nb = nbf.v4.new_notebook()
cells = []

def add_md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))

def add_code(code):
    cells.append(nbf.v4.new_code_cell(code.strip()))

# Title
add_md("""# 🛡️ VisionGuard Lite: Uncertainty-Aware Fall Detection with Evidence-Grounded SLM Explanations
### *An Edge-Deployable, Calibrated Skeleton Human Activity Recognition (HAR) Framework*

---
**Key System Pillars:**
1. **Lightweight Temporal Modeling**: TemporalCNN (Model B) achieving high recall and F1 on held-out subjects with only 66k parameters.
2. **Post-Hoc Probability Calibration**: Temperature scaling optimizing Expected Calibration Error (ECE) on validation logits.
3. **Selective Classification & Rejection Policy**: Flags ambiguous or low-visibility frames as UNCERTAIN to prevent false alarms.
4. **Evidence-Grounded Explanations**: Dual-engine generation (Deterministic Template + Quantized Local SLM) verified by an automated safety checker.
5. **Edge Optimization**: Exported to ONNX (16.3 KB), achieving <0.5 ms CPU latency for real-time mobile deployment.
""")

# Phase 1
add_md("""## Phase 1 — Environment & Project Setup
Inspect the runtime environment (CPU, RAM, GPU, disk) and set up the modular directory structure.
""")
add_code("""import sys, os, platform, torch, shutil

print('Python Version:', sys.version)
print('Platform:', platform.platform())
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name(0))

# Project root path (configured directly on Google Drive)
PROJECT_DIR = '/content/drive/MyDrive/VisionGuardLite'
if not os.path.exists(PROJECT_DIR):
    PROJECT_DIR = '/content/VisionGuardLite'
sys.path.insert(0, PROJECT_DIR)
print('Active Project Path:', PROJECT_DIR)
""")

# Phase 2
add_md("""## Phase 2 — Dataset Inspection & Schema Definitions
Defines activity categories, COCO-17 and MediaPipe-33 landmark topologies, and SAFER-Activities schema mappings.
""")
add_code("""from src.data.schema import ActivityClass, ACTIVITY_NAMES, COCO_17_LANDMARKS, MEDIAPIPE_33_LANDMARKS

print('Supported Activity Classes:')
for code, name in ACTIVITY_NAMES.items():
    print(f'  [{code}] {name.upper()}')

print(f'\\nCOCO 17 Keypoints ({len(COCO_17_LANDMARKS)}):', COCO_17_LANDMARKS[:6], '...')
print(f'MediaPipe 33 Keypoints ({len(MEDIAPIPE_33_LANDMARKS)}):', MEDIAPIPE_33_LANDMARKS[:6], '...')
""")

# Phase 3
add_md("""## Phase 3 — Data Preprocessing & Subject-Wise Partitioning
Applies person-centering (translating hips to origin), torso-scale normalization, missing joint imputation, and sliding window construction (30 frames @ 10 FPS = 3.0 seconds).
Ensures zero participant leakage by splitting data strictly by Subject ID.
""")
add_code("""from src.data.dataset import get_dataloaders
import numpy as np

data_dir = f'{PROJECT_DIR}/datasets/processed/coco_17'
train_l, val_l, test_l, weights = get_dataloaders(data_dir=data_dir, batch_size=32)

print('DataLoaders loaded successfully:')
print(f'  Train Windows: {len(train_l.dataset)}')
print(f'  Val Windows:   {len(val_l.dataset)} (Subjects P09, P10)')
print(f'  Test Windows:  {len(test_l.dataset)} (Held-out Subjects P11, P12)')
print(f'  Class Weights: {weights.tolist()}')
""")

# Phase 4
add_md("""## Phase 4 — Model Training & Comparative Evaluation
Compares Model A (BaselineMLP), Model B (TemporalCNN), and Model C (SmallTCN) on held-out test subjects.
""")
add_code("""import json

results_file = f'{PROJECT_DIR}/results/metrics/model_comparison.json'
with open(results_file, 'r') as f:
    results = json.load(f)

print(f'{\"Model\":<24} | {\"Params\":<8} | {\"Size (KB)\":<10} | {\"Precision\":<10} | {\"Recall\":<8} | {\"F1\":<8} | {\"AUROC\":<8} | {\"Latency (ms)\":<12}')
print('-' * 105)
for m_name, metrics in results.items():
    print(f\"{m_name:<24} | {metrics['parameters']:<8} | {metrics['model_size_kb']:<10} | {metrics['precision']:<10.4f} | {metrics['recall']:<8.4f} | {metrics['f1_score']:<8.4f} | {metrics['auroc']:<8.4f} | {metrics['median_latency_ms']:<12.3f}\")
""")

# Phase 5
add_md("""## Phase 5 — Uncertainty Calibration & Distribution Shift Resilience
Post-hoc probability calibration with Temperature Scaling, and stress testing against joint occlusions, noise, and scaling.
""")
add_code("""import json

calib_file = f'{PROJECT_DIR}/results/metrics/calibration_results.json'
with open(calib_file, 'r') as f:
    calib = json.load(f)

print('Probability Calibration Results (Model B):')
print(f\"  Optimal Temperature (T):   {calib['optimal_temperature']}\")
print(f\"  Test ECE Before:           {calib['test_ece_before']:.4f} ({calib['test_ece_before']*100:.2f}%)\")
print(f\"  Test ECE After:            {calib['test_ece_after']:.4f} ({calib['test_ece_after']*100:.2f}%)\")

shift_file = f'{PROJECT_DIR}/results/metrics/distribution_shift_results.json'
with open(shift_file, 'r') as f:
    shifts = json.load(f)

print('\\nDistribution Shift Stress-Testing:')
for shift, s_met in shifts.items():
    print(f\"  {shift:<22}: F1={s_met['f1_score']:.4f} | Recall={s_met['recall']:.4f} | AUROC={s_met['auroc']:.4f}\")
""")

# Phase 6
add_md("""## Phase 6 — Evidence-Grounded SLM Explanations & Automated Safety Verification
Dual-engine explanation system with automated safety checking against unsupported medical claims.
""")
add_code("""import json
from src.inference.slm_explainer import TemplateExplainer, EvidenceConsistencyChecker

explainer = TemplateExplainer()

evidence_example = {
    'detected_activity': 'fall',
    'confidence': 0.94,
    'landmark_visibility': 0.88,
    'temporal_consistency': 'high',
    'uncertain': False
}

explanation = explainer.explain(evidence_example)
safety = EvidenceConsistencyChecker.validate(evidence_example, explanation['explanation'])

print('Input Sensor Evidence:', json.dumps(evidence_example, indent=2))
print('\\nGenerated Explanation:', explanation['explanation'])
print('Safety Verification Passed:', safety['passed'])
""")

# Phase 7
add_md("""## Phase 7 — Interactive Gradio Demonstration
End-to-end video pose extraction, classification timeline plotting, and explanation.
""")
add_code("""from demo.app import analyze_video
import os

sample_video = f'{PROJECT_DIR}/demo/sample_activity.mp4'
if os.path.exists(sample_video):
    out_vid, plot_img, expl, telemetry = analyze_video(sample_video)
    print('Analysis Completed!')
    print('Decision:', telemetry['Status Decision'])
    print('Peak Fall Probability:', telemetry['Peak Fall Probability'])
    print('Per-Window Latency:', telemetry['Per-Window Inference Latency'])
    print('Explanation:', expl)
""")

# Phase 8
add_md("""## Phase 8 — Edge Optimization & ONNX Benchmarking
Verifies ONNX export numerical parity and measures batch-size-1 CPU latency.
""")
add_code("""import json

onnx_file = f'{PROJECT_DIR}/results/metrics/onnx_benchmark.json'
with open(onnx_file, 'r') as f:
    onnx_data = json.load(f)

print('ONNX Edge Deployment Benchmark:')
print(f\"  Model File:             {onnx_data['onnx_path']}\")
print(f\"  Model Size:             {onnx_data['model_size_kb']} KB\")
print(f\"  Max Parity Difference:  {onnx_data['max_absolute_diff']:.8f}\")
print(f\"  Numerical Parity OK:    {onnx_data['numerical_parity_passed']}\")
print(f\"  Median CPU Latency:     {onnx_data['median_latency_ms']:.3f} ms\")
print(f\"  P95 CPU Latency:        {onnx_data['p95_latency_ms']:.3f} ms\")
""")

# Phase 9
add_md("""## Phase 9 — Research Findings & Comparative Analysis
Visualizes loss curves, calibration reliability diagrams, and degradation profiles.
""")
add_code("""import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plot_paths = [
    f'{PROJECT_DIR}/results/plots/training_loss_curves.png',
    f'{PROJECT_DIR}/results/plots/reliability_diagram.png',
    f'{PROJECT_DIR}/results/plots/distribution_shift_degradation.png'
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, p in zip(axes, plot_paths):
    if os.path.exists(p):
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(os.path.basename(p))
plt.tight_layout()
plt.show()
""")

# Phase 10
add_md("""## Phase 10 — Drive Persistence & Verification
All model checkpoints, exported ONNX models, evaluation metrics, and notebooks are permanently saved in Google Drive.
""")
add_code("""print('Listing items in Google Drive project directory:')
for root, dirs, files in os.walk(PROJECT_DIR):
    rel = os.path.relpath(root, PROJECT_DIR)
    if rel == '.':
        print(f'Drive Root: {PROJECT_DIR}')
    else:
        indent = '  ' * rel.count(os.sep)
        print(f'{indent}📁 {os.path.basename(root)}/ ({len(files)} files)')
""")

nb.cells = cells

# Save notebook to multiple locations
target_paths = [
    f'{PROJECT_DIR}/VisionGuard_Lite_Colab.ipynb',
    f'{PROJECT_DIR}/notebooks/VisionGuard_Lite_Colab.ipynb',
    '/content/VisionGuard_Lite_Colab.ipynb'
]

for tp in target_paths:
    os.makedirs(os.path.dirname(os.path.abspath(tp)), exist_ok=True)
    with open(tp, 'w') as f:
        nbf.write(nb, f)
    print(f'Successfully wrote notebook to {tp}')
