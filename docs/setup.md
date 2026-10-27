# Setup and troubleshooting

## Requirements

Use Python 3.10 or later and an isolated virtual environment. The repository's CI matrix covers Python 3.10 and 3.11 on Linux and Windows; this is distinct from the broader Python version range in package metadata. macOS launch instructions are provided, but the CI workflow does not test macOS.

Install runtime dependencies with `python -m pip install -r requirements-local.txt`. This file includes the PyTorch CPU package index. For development, use `requirements-dev.txt`, which also installs pytest and coverage tooling.

Install FFmpeg for conversion of videos that a browser cannot play. On Debian/Ubuntu, the graphics dependencies used by CI can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y libgles2 libegl1 libgl1 libglib2.0-0 ffmpeg
```

## Required model assets

The demo requires these repository files and stops with an error if the real classifier artifacts are missing:

| Asset | Purpose |
| --- | --- |
| `checkpoints/pose_landmarker_lite.task` | MediaPipe landmark extraction |
| `checkpoints/real_temporalcnn_best.pt` | Real-data TemporalCNN weights and calibration metadata |
| `exported_models/real_temporalcnn.onnx` | Default CPU classifier |

Both classifier backends load at startup. The demo does not silently replace missing real-data weights with synthetic weights. Load model checkpoints only from a trusted source.

## Launch options

From the repository root:

```bash
./scripts/run_local.sh --port 7860
```

On Windows, use `.\scripts\run_local.ps1 -Port 7860`; `-HostAddress` changes the binding. For the Python launcher, `python demo/app.py --host 127.0.0.1 --port 7860` is equivalent, and `--share` explicitly enables a Gradio public link. The default host is `127.0.0.1`. Only expose the app when you intend to make uploaded video accessible to that host or service.

The launcher reads command-line arguments. `.env.example` describes reference values; the application does not automatically apply that file as configuration.

## Dataset utility

SAFER-Activities is optional and requires access approval from its maintainers. Set `HF_TOKEN` in your environment, then run:

```bash
python scripts/download_real_safer.py
```

The default destination is `datasets/raw/safer_real/` under this checkout. Use `--output_dir` to select another destination. Do not commit access tokens or raw human recordings. See [dataset provenance](datasets.md).

## Common failures

| Symptom | Action |
| --- | --- |
| EGL/GLES import failure | Install the Linux graphics libraries listed above. |
| Missing checkpoint or ONNX model | Restore the required trusted artifacts from the repository or regenerate them using the research workflows. |
| Port already in use | Choose another port, such as `--port 7861`. |
| Video cannot play in the browser | Install FFmpeg and use H.264 MP4 or WebM. |
| Uncertain result with poor tracking | Use a clearer, well-lit view of one person with less occlusion. High model probability alone does not establish a fall. |
| Clip too short | Supply at least ten sampled frames; longer clips provide more temporal context. |
| SAFER download denied | Obtain dataset access and supply a valid read token through the environment. |

## Repository conventions

Application orchestration lives in `demo/app.py`; presentation belongs in `demo/ui.py` and `demo/style.css`. Reusable computation belongs in the matching `src/` module. Keep model-artifact paths stable unless all runtime and experiment consumers are migrated together.

Place reusable developer commands in `scripts/`, documentation in `docs/`, images in `docs/assets/`, and dated evidence in `docs/reports/`. Experimental workflows and recorded outputs belong in `experiments/` and `results/`. Keep credentials, virtual environments, caches, generated demo outputs, and raw benchmark recordings out of Git.
