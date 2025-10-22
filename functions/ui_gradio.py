# ui_gradio.py
import gradio as gr
from functions.config import AppConfig
from pathlib import Path
import cv2
from functions.frame_bus import BUS
import numpy as np
from functions.paths import APP_DIR, resource_path


try:
    from pygrabber.dshow_graph import FilterGraph
    _HAS_PYGRABBER = True
except Exception:
    _HAS_PYGRABBER = False

def list_model_files(models_dir: Path) -> list[tuple[str, str]]:
    """
    Returns list of (label, value) pairs for Dropdown.
    label: filename; value: full path str
    """
    models = []
    for p in sorted(models_dir.glob("*.pth")):
        models.append((p.name, str(p)))
    return models


def list_cameras(max_index: int = 15):
    """
    Return [(label, index), ...] with names when possible.
    Uses DirectShow (pygrabber) to list friendly names and maps them
    to OpenCV indices by trying CAP_DSHOW in order.
    Falls back to probing if pygrabber is unavailable.
    """
    cams: list[tuple[str, int]] = []

    if _HAS_PYGRABBER:
        # Get names from DirectShow
        names = FilterGraph().get_input_devices()
        # Try to open by index with CAP_DSHOW and pair name->index by order
        for i, name in enumerate(names):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    label = f"{name}"
                else:
                    label = name
                cams.append((label, i))
            cap.release()

        # if nothing opened via DSHOW (drivers/backends), fall back to probe
        if cams:
            return cams

    # Fallback: generic OpenCV probe (no pretty names)
    for i in range(max_index):
        cap = cv2.VideoCapture(i)  # default backend
        ok = cap.isOpened()
        if not ok:
            cap.release()
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)  # Windows fallback
            ok = cap.isOpened()
        if ok:
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                label = f"Cam {i} ({w}x{h})"
            else:
                label = f"Cam {i} (available)"
            cams.append((label, i))
        cap.release()
    return cams


def clear_live_image():
    return gr.update(value=None)  # clears the image



def launch_ui(cfg: AppConfig, host: str = "127.0.0.1", port: int = 7860):
    with gr.Blocks(title="🎛 Control Panel") as demo:
        gr.Markdown("## 🎛 Control Panel")
        gr.Markdown("### Live View")
        live_img = gr.Image(label=None, show_label=False)

        # Timer callback: pull the latest frame from the bus.
        def pull_latest():
            return BUS.pull()

        # Update timer. Adjust if you want.

        t = gr.Timer(0.016)                  
        t.tick(pull_latest, outputs=live_img)

        


        # Model & IO
        with gr.Row():
            # Model selection from ./models
            models_dir = resource_path("functions", "models")
            model_choices = list_model_files(models_dir) 

            # pick emonet_8.pth if it exists; otherwise keep the first item
            default_model_path = ""
            if model_choices:
                default_model_path = next(
                    (v for (lbl, v) in model_choices if Path(v).name.lower() == "emonet_8.pth"),
                    model_choices[0][1]
                )

            model_select = gr.Dropdown(
                choices=model_choices,
                value=default_model_path, 
                label="Model selection",
                allow_custom_value=False,
            )


            # Camera selection by probing
            cam_choices = list_cameras(max_index=1)  # tweak max if needed
            camera_select = gr.Dropdown(
                choices=cam_choices,
                value=(cam_choices[0][1] if cam_choices else None),
                label="Camera selection",
                allow_custom_value=False,
            )

            #output_path = gr.Textbox(value=cfg.output_path, label="output_path (without .mp4, empty = off)")

            # Apply for model / camera / recording path
            
            apply = gr.Button("Apply Model/IO")
            apply.click(
                lambda mp, ci: cfg.update(model_path=(mp or ""), camera_index=int(ci)),
                inputs=[model_select, camera_select],
                outputs=[],
            )
            
        with gr.Accordion("Comparison model", open=False):

            models_dir = resource_path("functions", "models")
            cmp_choices = list_model_files(models_dir)

            compare_enable = gr.Checkbox(value=cfg.compare_enabled, label="Enable comparison model")
            compare_model_select = gr.Dropdown(
                choices=cmp_choices,
                value=(cfg.compare_model_path if cfg.compare_model_path else (cmp_choices[0][1] if cmp_choices else "")),
                label="Select comparison weights (.pth)",
                allow_custom_value=False,
            )
            apply_compare = gr.Button("Apply comparison model", scale = 0)

            # live toggle
            compare_enable.change(lambda v: cfg.update(compare_enabled=bool(v)), inputs=compare_enable, outputs=[])

            # apply
            apply_compare.click(
                lambda p: cfg.update(compare_model_path=(p or "").strip()),
                inputs=compare_model_select,
                outputs=[],
            )
           

        



        
        gr.Markdown("### Overlays (live)")
        with gr.Row():
            check_all_btn = gr.Button("✅ Check All", scale=0)
            uncheck_all_btn = gr.Button("🚫 Uncheck All", scale=0)


        # Group 1: Basic overlay
        with gr.Group():
            gr.Markdown("#### Basic visualizations")
            with gr.Row():
                show_bbox = gr.Checkbox(value=cfg.show_bbox, label="Bounding Box", scale = 0)
                label_only = gr.Checkbox(value=cfg.label_only, label="Emotions label", scale = 0)
                show_emoji = gr.Checkbox(value=cfg.show_emoji, label="Show emoji", scale = 0)
                show_face = gr.Checkbox(value=cfg.show_face, label="Face Landmarks", scale = 0)
                show_va   = gr.Checkbox(value=cfg.show_va,   label="Valence–Arousal Circumplex")
                



        #  Group 2: Bars overlay (parent + children)
        with gr.Group():
            gr.Markdown("#### Bars overlay")
            with gr.Row():
                bars_with_titles_only = gr.Checkbox(value=cfg.bars_with_titles_only, label="Progression Bar", scale = 0)
                bars_labels_only      = gr.Checkbox(value=cfg.bars_labels_only,      label="Bars_labels")
                #bars_around_bbox      = gr.Checkbox(value=cfg.bars_around_bbox,      label="bars_around_bbox")
                
                
            
            

        # Group 3: Extra visualizations
        with gr.Group():
            gr.Markdown("#### Extra visualizations")
            with gr.Row():

                viz_progress = gr.Checkbox(value=cfg.viz_progress, label="VA progress",scale = 0)  # just a label change
                viz_probs    = gr.Checkbox(value=cfg.viz_probs,    label="Emotion probabilities")

        

        show_emoji.change(lambda v: cfg.update(show_emoji=bool(v)), inputs=show_emoji, outputs=[])
        show_bbox.change(lambda v: cfg.update(show_bbox=bool(v)), inputs=show_bbox, outputs=[])
        label_only.change(lambda v: cfg.update(label_only=bool(v)), inputs=label_only, outputs=[])
        show_face.change(lambda v: cfg.update(show_face=bool(v)), inputs=show_face, outputs=[])
        show_va.change(lambda v: cfg.update(show_va=bool(v)),     inputs=show_va,   outputs=[])
        bars_with_titles_only.change(lambda v: cfg.update(bars_with_titles_only=bool(v)), inputs=bars_with_titles_only, outputs=[])
        bars_labels_only.change(lambda v: cfg.update(bars_labels_only=bool(v)),inputs=bars_labels_only,outputs=[])
        #bars_around_bbox.change( lambda v: cfg.update(bars_around_bbox=bool(v)), inputs=bars_around_bbox,outputs=[])
        viz_progress.change(lambda v: cfg.update(viz_progress=bool(v)), inputs=viz_progress, outputs=[])
        viz_probs.change(   lambda v: cfg.update(viz_probs=bool(v)),    inputs=viz_probs,    outputs=[])

        all_overlays = [
    show_bbox, label_only,
    bars_with_titles_only, bars_labels_only,
    show_face, show_va, viz_progress, viz_probs, show_emoji,
]
        check_all_btn.click(
            lambda: [True] * len(all_overlays),
            inputs=[],
            outputs=all_overlays,
            show_progress="hidden",
        )


        uncheck_all_btn.click(
            lambda: [False] * len(all_overlays),
            inputs=[],
            outputs=all_overlays,
            show_progress="hidden",
        )

        gr.Markdown("---")
        with gr.Row():
            start_cap = gr.Button("Start Capture",scale = 0)
            quit_cap = gr.Button("🛑 Quit Capture", scale = 0)

        start_cap.click(lambda: cfg.request_start_capture(), inputs=None, outputs=None)
        def on_quit():
            BUS.set_running(False)
            BUS.clear()
            #signal the loop to stop
            cfg.request_stop_capture()
            #return a clear for the component
            return gr.update(value=None)

        quit_cap.click(on_quit, inputs=None, outputs=live_img)

    demo.launch(
        server_name=host,
        server_port=port,
        show_api=False,
        inbrowser=False,
        quiet=True
    )   