"""
VisionGuard Lite - Comprehensive CPU Benchmarking Suite
Measures stage-by-stage latencies, throughput, peak memory, and CPU utilization
on consumer CPU hardware with Batch Size = 1.
"""

import os
import sys
import time
import json
import psutil
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.temporal_cnn import TemporalCNN
from src.data.pose_schema_audit import PoseSchemaAuditor
from src.data.preprocessing import PosePreprocessor
from src.evaluation.uncertainty import TemperatureScaler, UncertaintyPolicy
from src.inference.mediapipe_extractor import MediaPipePoseExtractor
from src.inference.slm_explainer import TemplateExplainer

def get_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)

def benchmark_cpu_suite(
    video_path: str = "demo/sample_activity.mp4",
    num_runs: int = 100
) -> Dict[str, Any]:
    print("=" * 65)
    print("   VisionGuard Lite — Consumer Hardware CPU Benchmark Suite")
    print(f"   OS: {sys.platform} | Python: {sys.version.split()[0]} | Batch Size: 1")
    print("=" * 65)

    results = {}
    mem_start = get_process_memory_mb()

    # 1. Model Loading Benchmark
    print("\n[1/8] Benchmarking Model Loading Time...")
    t0 = time.perf_counter()
    py_model = TemporalCNN(in_channels=51, num_classes=2)
    ckpt_path = REPO_ROOT / "checkpoints/real_temporalcnn_best.pt"
    if not ckpt_path.exists():
        ckpt_path = REPO_ROOT / "checkpoints/temporal_cnn_best.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    py_model.load_state_dict(state)
    py_model.eval()
    py_load_time_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    onnx_path = REPO_ROOT / "exported_models/real_temporalcnn.onnx"
    if not onnx_path.exists():
        onnx_path = REPO_ROOT / "exported_models/temporal_cnn_clean.onnx"
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    ort_session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])
    onnx_load_time_ms = (time.perf_counter() - t0) * 1000.0

    results["model_loading"] = {
        "pytorch_load_time_ms": round(py_load_time_ms, 2),
        "onnx_load_time_ms": round(onnx_load_time_ms, 2),
        "onnx_model_path": str(onnx_path.relative_to(REPO_ROOT)),
        "pytorch_ckpt_path": str(ckpt_path.relative_to(REPO_ROOT))
    }
    print(f"  PyTorch Load Time: {py_load_time_ms:.2f} ms")
    print(f"  ONNX Session Init: {onnx_load_time_ms:.2f} ms")

    # 2. Pose Detection Latency (MediaPipe)
    print("\n[2/8] Benchmarking Pose Detection (MediaPipe Tasks)...")
    extractor = MediaPipePoseExtractor()
    extracted = extractor.extract_from_video(video_path, max_frames=60, target_fps=10.0)
    extractor.close()
    
    annotated_frames = extracted["annotated_frames"]
    frame_count = max(1, extracted["frame_count"])
    landmarks = extracted["landmarks"] # (T, 33, 4)

    # Warmup and individual frame detection profiling
    det_times_ms = []
    extractor_single = MediaPipePoseExtractor()
    # Profile 30 individual frame inferences
    sample_bgr = annotated_frames[0]
    import mediapipe as mp
    import cv2
    rgb_frame = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    for _ in range(5):
        _ = extractor_single.detector.detect(mp_img) # warmup
    for _ in range(30):
        t0 = time.perf_counter()
        _ = extractor_single.detector.detect(mp_img)
        det_times_ms.append((time.perf_counter() - t0) * 1000.0)
    extractor_single.close()

    results["pose_detection"] = {
        "median_ms": round(float(np.median(det_times_ms)), 2),
        "p95_ms": round(float(np.percentile(det_times_ms, 95)), 2),
        "mean_ms": round(float(np.mean(det_times_ms)), 2),
        "min_ms": round(float(np.min(det_times_ms)), 2),
        "max_ms": round(float(np.max(det_times_ms)), 2)
    }
    print(f"  Median Latency: {results['pose_detection']['median_ms']:.2f} ms | P95: {results['pose_detection']['p95_ms']:.2f} ms")

    # 3. Feature Preprocessing Latency
    print("\n[3/8] Benchmarking Feature Preprocessing...")
    prep = PosePreprocessor(target_window_size=30)
    coco_kps = PoseSchemaAuditor.mediapipe_to_coco17(landmarks)
    prep_times_ms = []
    dummy_kps_window = coco_kps[:30] if len(coco_kps) >= 30 else np.pad(coco_kps, ((0, 30 - len(coco_kps)), (0, 0), (0, 0)))

    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = prep.normalize_skeleton(dummy_kps_window)
        prep_times_ms.append((time.perf_counter() - t0) * 1000.0)

    results["feature_preprocessing"] = {
        "median_ms": round(float(np.median(prep_times_ms)), 4),
        "p95_ms": round(float(np.percentile(prep_times_ms, 95)), 4),
        "mean_ms": round(float(np.mean(prep_times_ms)), 4)
    }
    print(f"  Median Latency: {results['feature_preprocessing']['median_ms']:.4f} ms")

    # 4. Classifier Inference Latency (Batch Size 1: PyTorch vs ONNX Runtime)
    print(f"\n[4/8] Benchmarking Classifier Inference ({num_runs} iterations, Batch Size = 1)...")
    dummy_input_np = np.random.randn(1, 51, 30).astype(np.float32)
    dummy_input_torch = torch.from_numpy(dummy_input_np)

    # PyTorch inference
    py_times_ms = []
    with torch.no_grad():
        for _ in range(10):
            _ = py_model(dummy_input_torch)
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = py_model(dummy_input_torch)
            py_times_ms.append((time.perf_counter() - t0) * 1000.0)

    # ONNX Runtime inference
    onnx_times_ms = []
    input_name = ort_session.get_inputs()[0].name
    for _ in range(10):
        _ = ort_session.run(None, {input_name: dummy_input_np})
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = ort_session.run(None, {input_name: dummy_input_np})
        onnx_times_ms.append((time.perf_counter() - t0) * 1000.0)

    # Parity check
    with torch.no_grad():
        torch_logits = py_model(dummy_input_torch).numpy()
    onnx_logits = ort_session.run(None, {input_name: dummy_input_np})[0]
    max_parity_diff = float(np.max(np.abs(torch_logits - onnx_logits)))

    results["classifier_inference"] = {
        "batch_size": 1,
        "pytorch": {
            "median_ms": round(float(np.median(py_times_ms)), 4),
            "p95_ms": round(float(np.percentile(py_times_ms, 95)), 4),
            "mean_ms": round(float(np.mean(py_times_ms)), 4)
        },
        "onnx_runtime": {
            "median_ms": round(float(np.median(onnx_times_ms)), 4),
            "p95_ms": round(float(np.percentile(onnx_times_ms, 95)), 4),
            "mean_ms": round(float(np.mean(onnx_times_ms)), 4)
        },
        "numerical_parity_max_diff": max_parity_diff,
        "onnx_speedup_factor": round(float(np.median(py_times_ms) / max(0.0001, np.median(onnx_times_ms))), 2)
    }
    print(f"  PyTorch Median: {results['classifier_inference']['pytorch']['median_ms']:.4f} ms")
    print(f"  ONNX Median:    {results['classifier_inference']['onnx_runtime']['median_ms']:.4f} ms")
    print(f"  Max Parity Diff: {max_parity_diff:.2e}")

    # 5. Uncertainty Estimation Benchmark
    print("\n[5/8] Benchmarking Uncertainty Estimation...")
    policy = UncertaintyPolicy(confidence_threshold=0.70, max_entropy=0.50)
    sample_probs = np.array([[0.82, 0.18]], dtype=np.float32)
    unc_times_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = policy.predict_with_rejection(sample_probs)
        unc_times_ms.append((time.perf_counter() - t0) * 1000.0)

    results["uncertainty_estimation"] = {
        "median_ms": round(float(np.median(unc_times_ms)), 4),
        "p95_ms": round(float(np.percentile(unc_times_ms, 95)), 4)
    }
    print(f"  Median Latency: {results['uncertainty_estimation']['median_ms']:.4f} ms")

    # 6. Temporal Filtering / Confirmation Benchmark
    print("\n[6/8] Benchmarking Temporal Confirmation Filtering...")
    filter_times_ms = []
    window_preds = [0, 0, 1, 1, 1, 0, 1, 0]
    for _ in range(num_runs):
        t0 = time.perf_counter()
        confirmed = []
        counter = 0
        for p in window_preds:
            if p == 1:
                counter += 1
            else:
                counter = 0
            confirmed.append(1 if counter >= 2 else 0)
        filter_times_ms.append((time.perf_counter() - t0) * 1000.0)

    results["temporal_filtering"] = {
        "median_ms": round(float(np.median(filter_times_ms)), 5),
        "p95_ms": round(float(np.percentile(filter_times_ms, 95)), 5)
    }
    print(f"  Median Latency: {results['temporal_filtering']['median_ms']:.5f} ms")

    # 7. Optional SLM Inference Benchmark
    print("\n[7/8] Benchmarking SLM / Explanation Generation...")
    explainer = TemplateExplainer()
    sample_evidence = {
        "detected_activity": "fall",
        "confidence": 0.88,
        "landmark_visibility": 0.92,
        "temporal_consistency": "high",
        "uncertain": False
    }
    slm_times_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = explainer.explain(sample_evidence)
        slm_times_ms.append((time.perf_counter() - t0) * 1000.0)

    results["slm_explanation"] = {
        "type": "TemplateExplainer (Deterministic, Zero GPU overhead)",
        "median_ms": round(float(np.median(slm_times_ms)), 4),
        "p95_ms": round(float(np.percentile(slm_times_ms, 95)), 4),
        "notes": "SLM transformer generation kept optional for low-resource CPU laptops."
    }
    print(f"  Median Latency: {results['slm_explanation']['median_ms']:.4f} ms")

    # 8. Complete End-to-End Video Processing & Resource Telemetry
    print(f"\n[8/8] Benchmarking Complete Video Processing on '{video_path}'...")
    cpu_percent_start = psutil.cpu_percent(interval=None)
    t_vid_start = time.perf_counter()
    
    # Process full video pipeline
    extractor_full = MediaPipePoseExtractor()
    extracted_full = extractor_full.extract_from_video(video_path, max_frames=120, target_fps=10.0)
    extractor_full.close()

    raw_lms_full = extracted_full["landmarks"]
    n_frames = extracted_full["frame_count"]
    duration_sec = extracted_full["duration_sec"]
    
    coco_kps_full = PoseSchemaAuditor.mediapipe_to_coco17(raw_lms_full)
    norm_wins = []
    if len(coco_kps_full) >= 30:
        for s in range(0, len(coco_kps_full) - 30 + 1, 5):
            norm_wins.append(prep.normalize_skeleton(coco_kps_full[s:s+30]))
    else:
        pad = np.repeat(coco_kps_full[-1:], 30 - len(coco_kps_full), axis=0)
        norm_wins.append(prep.normalize_skeleton(np.concatenate([coco_kps_full, pad], axis=0)))
    
    norm_wins = np.array(norm_wins, dtype=np.float32)
    W, T, J, C = norm_wins.shape
    in_batch = norm_wins.reshape(W, T, J * C).transpose(0, 2, 1)
    
    # Run through ONNX
    onnx_logits_full = ort_session.run(None, {input_name: in_batch})[0]
    calib_temp = float(ckpt.get("temperature", 1.627))
    probs_full = torch.softmax(torch.tensor(onnx_logits_full) / calib_temp, dim=1)[:, 1].numpy()
    
    t_vid_elapsed = time.perf_counter() - t_vid_start
    cpu_percent_end = psutil.cpu_percent(interval=0.1)
    mem_peak = get_process_memory_mb()

    fps = n_frames / max(0.001, t_vid_elapsed)

    results["end_to_end_video"] = {
        "video_duration_sec": round(duration_sec, 2),
        "frames_processed": n_frames,
        "windows_evaluated": W,
        "total_elapsed_sec": round(t_vid_elapsed, 3),
        "effective_fps": round(fps, 2),
        "realtime_factor": round(duration_sec / max(0.001, t_vid_elapsed), 2),
        "memory_start_mb": round(mem_start, 2),
        "memory_peak_mb": round(mem_peak, 2),
        "memory_delta_mb": round(mem_peak - mem_start, 2),
        "cpu_utilization_percent": round(cpu_percent_end, 1)
    }

    print(f"  Total Video Duration: {duration_sec:.1f} s | Frames: {n_frames}")
    print(f"  Total Processing Time: {t_vid_elapsed:.2f} s")
    print(f"  Effective Processing Speed: {fps:.1f} FPS (Real-time factor: {results['end_to_end_video']['realtime_factor']}x)")
    print(f"  Peak RAM: {mem_peak:.1f} MB (Delta: +{results['end_to_end_video']['memory_delta_mb']:.1f} MB)")

    # Save results to json
    out_dir = REPO_ROOT / "results/metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "cpu_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Benchmark results saved to: {out_json.relative_to(REPO_ROOT)}")
    return results

if __name__ == "__main__":
    benchmark_cpu_suite()
