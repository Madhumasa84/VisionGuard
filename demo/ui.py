"""Gradio presentation layer for the local video analysis workspace."""
from html import escape
from pathlib import Path

import gradio as gr


EMPTY_SUMMARY = """<div class="result-summary" role="status" aria-live="polite">
<span class="eyebrow">ANALYSIS STATUS</span><h2>Ready when you are</h2>
<p>Add a video or choose an example, then select Analyze video.</p></div>"""


def result_summary(metrics):
    """Render only escaped pipeline values into the results card."""
    status = metrics.get("Decision Status")
    if not status:
        return (f'<div class="result-summary" role="status" aria-live="polite">'
                f'<span class="eyebrow">ANALYSIS STATUS</span><h2>{escape(str(metrics.get("Status", "Unable to analyze")))}</h2>'
                '<p>Review the message below and try again.</p></div>')
    tone = "alert" if status == "CONFIRMED FALL EVENT" else "review" if status != "NORMAL ACTIVITY" else "normal"
    values = [("Peak fall probability", metrics["Peak Fall Probability"]),
              ("Pose visibility", metrics["Mean Landmark Visibility"]),
              ("Processing speed", metrics["Effective Processing FPS"])]
    cards = ''.join(f'<div><span>{label}</span><strong>{escape(str(value))}</strong></div>' for label, value in values)
    return (f'<div class="result-summary {tone}" role="status" aria-live="polite">'
            f'<span class="eyebrow">ANALYSIS COMPLETE</span><h2>{escape(status)}</h2>'
            f'<div class="metric-grid">{cards}</div></div>')


def interface_style():
    return dict(
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate", font=["Arial", "sans-serif"]).set(
            body_text_color_subdued="#475569", body_text_color_subdued_dark="#a8b7ca",
            block_label_text_color="#0f766e", block_label_text_color_dark="#5eead4",
            button_primary_background_fill="#0f766e", button_primary_background_fill_hover="#115e59",
            button_primary_background_fill_dark="#0f766e", button_primary_background_fill_hover_dark="#115e59",
            button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
        ),
        css=Path(__file__).with_name("style.css").read_text(),
    )


def build_interface(process_video, repo_root):
    def analyze(*args):
        video, timeline, explanation, metrics = process_video(*args)
        return video, timeline, explanation, metrics, result_summary(metrics)

    # Preserve compatibility with the declared Gradio >=4 dependency.
    style = interface_style() if int(gr.__version__.split('.')[0]) < 6 else {}
    with gr.Blocks(title="VisionGuard Lite | Video analysis", analytics_enabled=False,
                   delete_cache=(3600, 3600), **style) as app:
        gr.HTML('''<header class="app-header"><div class="brand"><span class="brand-mark" aria-hidden="true">V<span> /</span></span>
            <div>VisionGuard <span class="brand-lite">Lite</span><small>VIDEO ANALYSIS WORKSPACE</small></div></div>
            <span class="local-badge"><span aria-hidden="true">●</span> CPU processing</span></header>
            <section class="hero"><div><span class="eyebrow">POSE-BASED FALL DETECTION</span>
            <h1>A clearer view of movement.</h1><p>Turn a video into tracked motion, a probability timeline, and an evidence-based summary.</p></div>
            <div class="hero-note"><strong>Observe. Analyze. Review.</strong><span>Built for research. Designed for clarity.</span></div></section>''')
        gr.Markdown("**Research use only.** This model can miss falls and produce false alerts. "
                    "It is not a medical device or an emergency monitoring service.", elem_id="safety-notice")
        with gr.Row(equal_height=False, elem_id="workspace"):
            with gr.Column(scale=4, min_width=300, elem_classes=["workspace-panel"]):
                gr.Markdown("### 01 / Source video\nUpload a clip or record with your camera.")
                video_input = gr.Video(label="Source video", sources=["upload", "webcam"], height=260)
                gr.Markdown("Keep one person clearly visible. Analysis covers up to 250 sampled frames (about 25 seconds at 10 FPS).", elem_classes=["helper-copy"])
                with gr.Accordion("Analysis settings", open=False):
                    confirmation = gr.Slider(1, 4, value=2, step=1, label="Confirmation windows (K)",
                                             info="Consecutive positive windows needed for an alert. Default: 2.")
                    threshold = gr.Slider(0.20, 0.90, value=0.50, step=0.05, label="Fall probability threshold",
                                          info="Lower values trigger more positive windows. Default: 0.50.")
                    backend = gr.Dropdown(["ONNX Runtime (Fastest / Recommended)", "PyTorch (CPU)"],
                                          value="ONNX Runtime (Fastest / Recommended)", label="Processing engine")
                run = gr.Button("Analyze video", variant="primary", size="lg", elem_id="analyze-button")
                clear = gr.Button("Clear workspace", variant="secondary")
                examples = []
                for path in [repo_root / "demo/sample_activity.mp4",
                             repo_root / "experiments/gmdcsa24_real/data/raw/gmdcsa24/Subject 1/Fall/01.mp4"]:
                    if path.is_file():
                        examples.append([str(path)])
                if examples:
                    gr.Examples(examples=examples, inputs=[video_input], label="Try an example", examples_per_page=2)
            with gr.Column(scale=7, min_width=300, elem_classes=["workspace-panel"]):
                summary = gr.HTML(EMPTY_SUMMARY)
                with gr.Tabs():
                    with gr.Tab("Video review"):
                        annotated = gr.Video(label="Pose tracking & window decisions", interactive=False, height=330)
                        gr.Markdown("Banners show decisions as windows become available. The summary covers the entire analyzed clip.", elem_classes=["helper-copy"])
                    with gr.Tab("Probability timeline"):
                        timeline = gr.Image(label="Fall probability over time", interactive=False, type="filepath", elem_id="timeline-output")
                        gr.Markdown("Dashed line: your threshold. Shaded areas: windows confirmed by the selected policy.", elem_classes=["helper-copy"])
                explanation = gr.Textbox(label="Evidence summary", lines=3, interactive=False,
                                         placeholder="Your analysis summary will appear here.")
                with gr.Accordion("Performance & diagnostics", open=False):
                    telemetry = gr.JSON(label="Analysis details")
        with gr.Accordion("About the model & its limitations", open=False, elem_id="model-notes"):
            gr.Markdown("""**Model:** TemporalCNN trained on GMDCSA-24 staged human falls, evaluated using leave-one-subject-out validation across four participants. Inference runs on the server CPU.

**Interpretation:** A confirmed event means the model exceeded your threshold for K consecutive windows. It does not verify an injury. Low pose visibility makes predictions less reliable. Confirmation policy delay measures time from the start of the confirming positive run, not from the physical onset of a fall.

**Measured limitations:** The raw TemporalCNN benchmark reports approximately 119.72 false alarms/hour; K=2 reduces this to 95.00, with 92.2% event recall. These research results do not establish reliability in everyday environments or older populations.

**Privacy:** Uploaded clips and generated results are processed on the host running this app and stored temporarily. Webcam input records a clip for analysis; it is not continuous live monitoring.""")
        gr.HTML('<footer class="app-footer"><span>VisionGuard Lite</span><span>CPU-first vision · Human review required</span></footer>')
        inputs = [video_input, confirmation, threshold, backend]
        outputs = [annotated, timeline, explanation, telemetry, summary]
        analysis_event = run.click(analyze, inputs=inputs, outputs=outputs, concurrency_limit=1,
                                  concurrency_id="video-analysis", api_name="analyze_video")
        clear.click(lambda: (None, None, None, "", None, EMPTY_SUMMARY),
                    outputs=[video_input, *outputs], cancels=[analysis_event], queue=False, api_name=False)
        # Changing the source/settings invalidates the previous result.
        for component in inputs:
            component.change(lambda: (None, None, "", None, EMPTY_SUMMARY), outputs=outputs,
                             cancels=[analysis_event], queue=False, api_name=False)
    return app
