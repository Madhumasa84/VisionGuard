"""
VisionGuard Lite - Professional CPU-First Fall Detection Research Demonstration
Interactive Gradio Web Application:
- Real-time MediaPipe-33 pose extraction & COCO-17 landmark alignment
- Real GMDCSA-24 trained TemporalCNN (PyTorch and ONNX Runtime backends)
- Post-hoc Temperature Scaling and predictive uncertainty estimation
- Temporal confirmation filtering (raw vs K-window confirmed alerts)
- Grounded explanation engine with safety validation
- Interactive video player, probability trajectory timeline, and performance telemetry
"""

import os
import sys
import time
import argparse
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

# Portable project root discovery
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure matplotlib runs headlessly without X11
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

import cv2
import psutil
import numpy as np
import torch
import onnxruntime as ort
import gradio as gr

from src.models.temporal_cnn import TemporalCNN
from src.data.pose_schema_audit import PoseSchemaAuditor
from src.data.preprocessing import PosePreprocessor
from src.inference.mediapipe_extractor import MediaPipePoseExtractor
from src.inference.slm_explainer import TemplateExplainer, EvidenceConsistencyChecker

# Global Model & Engine Holders
class PipelineManager:
    def __init__(self):
        self.preprocessor = PosePreprocessor(target_window_size=30)
        self.explainer = TemplateExplainer()
        self.py_model = None
        self.onnx_session = None
        self.input_name = None
        self.temperature = 1.627
        self.optimal_threshold = 0.50
        self.active_backend = "onnx"
        self._load_models()

    def _load_models(self):
        # 1. Load Real PyTorch Checkpoint (Strictly real GMDCSA model, no silent synthetic fallback)
        ckpt_path = REPO_ROOT / "checkpoints/real_temporalcnn_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Real-data model checkpoint not found at '{ckpt_path}'. "
                "VisionGuard Lite strictly requires real GMDCSA-trained weights for deployment. "
                "Run 'python experiments/gmdcsa24_real/src/train_loso.py' to regenerate."
            )

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.temperature = float(ckpt.get("temperature", 1.627))
        self.optimal_threshold = float(ckpt.get("optimal_threshold", 0.50))
        self.py_model = TemporalCNN(in_channels=51, num_classes=2)
        state = ckpt.get("model_state", ckpt)
        self.py_model.load_state_dict(state)
        self.py_model.eval()

        # 2. Load Real ONNX Session
        onnx_path = REPO_ROOT / "exported_models/real_temporalcnn.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Real-data ONNX model not found at '{onnx_path}'. "
                "Export it using 'python -c \"from src.inference.onnx_exporter import export_temporal_cnn_to_onnx; ...\"'."
            )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        self.onnx_session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])
        self.input_name = self.onnx_session.get_inputs()[0].name

    def predict_window(self, window_51_30: np.ndarray, backend: str = "onnx") -> float:
        """Runs single-window inference (1, 51, 30) through chosen backend."""
        if backend == "onnx" and self.onnx_session is not None:
            logits = self.onnx_session.run(None, {self.input_name: window_51_30})[0]
            scaled = logits / max(0.1, self.temperature)
            prob = float(torch.softmax(torch.tensor(scaled), dim=1)[0, 1].item())
            return prob
        elif self.py_model is not None:
            with torch.no_grad():
                tensor_in = torch.from_numpy(window_51_30)
                logits = self.py_model(tensor_in).numpy()
                scaled = logits / max(0.1, self.temperature)
                prob = float(torch.softmax(torch.tensor(scaled), dim=1)[0, 1].item())
                return prob
        return 0.0

manager = PipelineManager()

def confirm_windows(probabilities, times, threshold, k):
    """Confirm consecutive positives; delay starts at the current positive run."""
    states = []
    run_start = None
    delay = None
    count = 0
    for probability, timestamp in zip(probabilities, times):
        if probability >= threshold:
            if count == 0:
                run_start = timestamp
            count += 1
        else:
            count = 0
            run_start = None
        confirmed = count >= k
        states.append(int(confirmed))
        if confirmed and delay is None:
            delay = round(timestamp - run_start, 2)
    return states, delay


def process_video_pipeline(
    video_path: Optional[str],
    confirmation_k: int = 2,
    confidence_thresh: float = 0.50,
    backend_choice: str = "ONNX Runtime (Fastest / Recommended)"
) -> Tuple[Optional[str], Optional[str], str, Dict[str, Any]]:
    """
    Executes end-to-end fall analysis pipeline:
    1. Input validation
    2. MediaPipe-33 landmark extraction
    3. COCO-17 anatomical alignment
    4. Windowing and skeleton normalization
    5. Calibrated inference with temporal confirmation
    6. Grounded explanation & telemetry generation
    """
    if not video_path or not os.path.isfile(video_path):
        return None, None, "[WARNING] Please upload a valid video file or record a clip via webcam.", {
            "Status": "Invalid Input",
            "Error": "No video file received"
        }

    if (isinstance(confirmation_k, bool) or not isinstance(confirmation_k, (int, float))
            or confirmation_k not in (1, 2, 3, 4)
            or not isinstance(confidence_thresh, (int, float))
            or not np.isfinite(confidence_thresh) or not 0.20 <= confidence_thresh <= 0.90
            or backend_choice not in ("ONNX Runtime (Fastest / Recommended)", "PyTorch (CPU)")):
        return None, None, "[WARNING] Choose valid analysis settings.", {"Status": "Invalid Settings"}
    confirmation_k = int(confirmation_k)
    t_start = time.perf_counter()
    backend = "onnx" if "ONNX" in backend_choice else "pytorch"

    # Step 1: MediaPipe Pose Extraction
    t_pose0 = time.perf_counter()
    extractor = None
    try:
        extractor = MediaPipePoseExtractor()
        extracted = extractor.extract_from_video(video_path, max_frames=250, target_fps=10.0)
    except Exception as e:
        return None, None, f"[ERROR] Pose extraction error: {str(e)}", {"Status": "Extraction Failure", "Error": str(e)}

    finally:
        if extractor is not None:
            extractor.close()

    t_pose_ms = (time.perf_counter() - t_pose0) * 1000.0
    raw_lms = extracted["landmarks"] # (T, 33, 4)
    annotated_frames = extracted["annotated_frames"]
    duration_sec = extracted["duration_sec"]
    num_frames = extracted["frame_count"]
    sampling_fps = extracted["fps"]

    if num_frames < 10:
        return None, None, "[WARNING] Video sequence is too short (< 10 frames). Please provide a longer recording with a clearly visible person.", {
            "Status": "Video Too Short",
            "Frames Detected": num_frames
        }

    # Step 2: Strict Schema Conversion (MediaPipe-33 -> COCO-17)
    coco_kps = PoseSchemaAuditor.mediapipe_to_coco17(raw_lms)
    mean_visibility = float(np.mean(coco_kps[:, :, 2]))

    # Step 3: Sliding Windows & Coordinate Normalization
    window_size = 30
    stride = 5 # 0.5s stride
    windows_51 = []
    window_times = []
    window_visibility = []

    if num_frames < window_size:
        pad_len = window_size - num_frames
        pad_frames = np.repeat(coco_kps[-1:], pad_len, axis=0)
        padded_kps = np.concatenate([coco_kps, pad_frames], axis=0)
        norm_w = manager.preprocessor.normalize_skeleton(padded_kps)
        w_51 = norm_w.reshape(1, window_size, 17 * 3).transpose(0, 2, 1)
        windows_51.append(w_51)
        window_visibility.append(mean_visibility)
        window_times.append(duration_sec)
    else:
        for s in range(0, num_frames - window_size + 1, stride):
            e = s + window_size
            norm_w = manager.preprocessor.normalize_skeleton(coco_kps[s:e])
            w_51 = norm_w.reshape(1, window_size, 17 * 3).transpose(0, 2, 1)
            windows_51.append(w_51)
            window_visibility.append(float(np.mean(coco_kps[s:e, :, 2])))
            window_times.append(e / sampling_fps)

    W = len(windows_51)
    if W == 0:
        return None, None, "[WARNING] No valid temporal windows could be extracted.", {"Status": "No Windows"}

    # Step 4: Batch / Single-Sample Inference
    t_infer0 = time.perf_counter()
    raw_probs = []
    for w_in in windows_51:
        p = manager.predict_window(w_in, backend=backend)
        raw_probs.append(p)
    t_infer_ms = (time.perf_counter() - t_infer0) * 1000.0 / max(1, W)

    # Step 5: Temporal Confirmation State Machine
    confirmed_state, confirmation_delay_sec = confirm_windows(
        [p if visibility >= 0.50 else -1.0 for p, visibility in zip(raw_probs, window_visibility)],
        window_times, confidence_thresh, confirmation_k
    )

    peak_prob = float(np.max(raw_probs))
    avg_prob = float(np.mean(raw_probs))
    has_confirmed_fall = any(s == 1 for s in confirmed_state)
    has_raw_spike = peak_prob >= confidence_thresh
    is_uncertain = (0.35 <= peak_prob < confidence_thresh) or (mean_visibility < 0.50)

    # State decision
    if mean_visibility < 0.50:
        final_state = "UNCERTAIN / LOW TRACKING CONFIDENCE"
        activity_label = "possible_fall"
    elif has_confirmed_fall:
        final_state = "CONFIRMED FALL EVENT"
        activity_label = "fall"
    elif has_raw_spike:
        final_state = "UNCONFIRMED SPIKE (Filtered by K-Window Policy)"
        activity_label = "possible_fall"
    elif is_uncertain:
        final_state = "UNCERTAIN / LOW TRACKING CONFIDENCE"
        activity_label = "possible_fall"
    else:
        final_state = "NORMAL ACTIVITY"
        activity_label = "normal"

    confidence_score = peak_prob if (has_confirmed_fall or has_raw_spike) else (1.0 - avg_prob)

    # Step 6: Generate Grounded Explanation
    evidence = {
        "detected_activity": activity_label,
        "confidence": round(confidence_score, 3),
        "landmark_visibility": round(mean_visibility, 3),
        "temporal_consistency": "high" if has_confirmed_fall else ("mixed" if has_raw_spike else "stable"),
        "uncertain": is_uncertain or (has_raw_spike and not has_confirmed_fall),
        "confirmation_windows": confirmation_k,
        "confirmed_by_policy": has_confirmed_fall
    }
    explanation_res = manager.explainer.explain(evidence)
    safety_check = EvidenceConsistencyChecker.validate(evidence, explanation_res["explanation"])

    # Step 7: Render Multi-Signal Timeline Visualization
    output_dir = Path(tempfile.mkdtemp(prefix="visionguard-"))
    fig = Figure(figsize=(9, 3.6), dpi=130, layout="constrained")
    ax = fig.subplots()
    ax.plot(window_times, raw_probs, 'b-o', linewidth=2.2, markersize=5, label='Calibrated Fall Probability (P)')
    
    # Fill confirmed region
    confirmed_np = np.array(confirmed_state)
    if np.any(confirmed_np == 1):
        ax.fill_between(window_times, 0, 1.05, where=(confirmed_np == 1), color='red', alpha=0.18, label=f'Confirmed Fall Zone (K={confirmation_k})')

    ax.axhline(confidence_thresh, color='crimson', linestyle='--', linewidth=1.5, label=f'Fall Threshold ({confidence_thresh:.2f})')
    ax.axhline(0.35, color='darkorange', linestyle=':', linewidth=1.4, label='Uncertainty Boundary (0.35)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(duration_sec, window_times[-1] if window_times else 1.0))
    ax.set_xlabel('Video Time (seconds)', fontsize=10)
    ax.set_ylabel('Fall Probability', fontsize=10)
    ax.set_title(f'Temporal Fall Trajectory (Model: GMDCSA-24 TemporalCNN | Backend: {backend.upper()})', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
    ax.grid(True, linestyle='--', alpha=0.4)

    timeline_img_path = str(output_dir / "timeline.png")
    fig.savefig(timeline_img_path)

    # Step 8: Render Output Annotated Video
    out_video_path = str(output_dir / "annotated.mp4")
    if len(annotated_frames) > 0:
        h, w, _ = annotated_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(out_video_path, fourcc, sampling_fps, (w, h))
        if not vw.isOpened():
            vw.release()
            return None, timeline_img_path, "[ERROR] Could not encode the annotated video.", {"Status": "Output Failure"}
        try:
            for idx, f in enumerate(annotated_frames):
                t_cur = (idx + 1) / sampling_fps
                available = [i for i, t in enumerate(window_times) if t <= t_cur]
                if not available:
                    frame_state, color = "COLLECTING POSE", (220, 220, 220)
                else:
                    i = available[-1]
                    if window_visibility[i] < 0.50:
                        frame_state, color = "UNCERTAIN TRACKING", (0, 165, 255)
                    elif confirmed_state[i]:
                        frame_state, color = "CONFIRMED FALL", (0, 0, 220)
                    elif raw_probs[i] >= confidence_thresh:
                        frame_state, color = "UNCONFIRMED SPIKE", (0, 140, 255)
                    elif mean_visibility < 0.50 or raw_probs[i] >= 0.35:
                        frame_state, color = "UNCERTAIN", (0, 165, 255)
                    else:
                        frame_state, color = "NORMAL ACTIVITY", (0, 200, 0)
                cv2.rectangle(f, (10, 10), (w - 10, 50), (25, 25, 25), -1)
                status_str = f"{frame_state} | T: {t_cur:.1f}s"
                cv2.putText(f, status_str, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                vw.write(f)
        finally:
            vw.release()
    else:
        return None, timeline_img_path, "[ERROR] No video frames available.", {"Status": "Output Failure"}

    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    effective_fps = num_frames / max(0.001, total_time_ms / 1000.0)
    mem_mb = psutil.Process().memory_info().rss / (1024.0 * 1024.0)

    metrics = {
        "Decision Status": final_state,
        "Active Classifier": f"TemporalCNN (GMDCSA-24 Real LOSO Fold 3)",
        "Inference Engine": f"{backend.upper()} (Consumer CPU)",
        "Peak Fall Probability": f"{peak_prob:.1%}",
        "Mean Landmark Visibility": f"{mean_visibility:.1%}",
        "Temporal Windows Analyzed": W,
        "Windows With Usable Tracking": sum(v >= 0.50 for v in window_visibility),
        "Analyzed Duration": f"{duration_sec:.1f} s (up to 250 sampled frames)",
        "Temporal Confirmation Setting": f"K = {confirmation_k} consecutive windows",
        "Confirmation Policy Delay": f"{confirmation_delay_sec} s" if confirmation_delay_sec is not None else "N/A (No sustained fall)",
        "Per-Window Inference Time": f"{t_infer_ms:.2f} ms",
        "Pose Extraction Time": f"{t_pose_ms:.1f} ms",
        "Total Pipeline Latency": f"{total_time_ms:.1f} ms",
        "Effective Processing FPS": f"{effective_fps:.1f} FPS",
        "RAM Usage": f"{mem_mb:.1f} MB",
        "Safety Verifier Passed": safety_check["passed"]
    }

    return out_video_path, timeline_img_path, explanation_res["explanation"], metrics

def build_app():
    from demo.ui import build_interface
    return build_interface(process_video_pipeline, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="VisionGuard Lite Web Application")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host IP")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", default=False, help="Create Gradio public link")
    args = parser.parse_args()

    demo = build_app()
    print(f"Starting VisionGuard Lite on http://{args.host}:{args.port} (share={args.share})...")
    # Gradio 6 moved theme and CSS from Blocks to launch.
    from demo.ui import interface_style
    style = interface_style() if int(gr.__version__.split(".")[0]) >= 6 else {}
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, **style)

if __name__ == "__main__":
    main()
