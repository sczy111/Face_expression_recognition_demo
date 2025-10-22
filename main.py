
import threading
import logging
import time
from functions.config import AppConfig
from emonet_core import EmoNetApp
from functions.ui_gradio import launch_ui
import ctypes
import sys
from functions.frame_bus import BUS
import os

def ui_thread(cfg: AppConfig):
    # COM init for DirectShow/pygrabber in this thread
    ctypes.windll.ole32.CoInitialize(None)
    try:
        launch_ui(cfg, host="127.0.0.1", port=7860)
    finally:
        ctypes.windll.ole32.CoUninitialize()
def shutdown_callback():
    log = logging.getLogger("MAIN")
    log.info("🌐 Web UI closed. Shutting down the program...")
    try:
        BUS.set_running(False)
        BUS.clear()
    except Exception:
        pass
    # Hard-exit the whole process (works even if this runs in a worker thread)
    os._exit(0)

if __name__ == "__main__":
    
    logging.getLogger("gradio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    log = logging.getLogger("MAIN")
    log.info("Launching...")

    cfg = AppConfig()
    app = EmoNetApp(cfg)

    
    t = threading.Thread(target=ui_thread, args=(cfg,), daemon=True)
    t.start()
    log.info("✅ Program started successfully. Open the UI in your browser: http://127.0.0.1:7860")
    #log.info("Waiting for 'Start Capture' command from UI")
    while True:
        if cfg.consume_start_request():
            log.info("Start Capture requested. Launching OpenCV loop")
            app.run_capture_loop()
            log.info("Capture loop ended. Waiting for next Start Capture")
        time.sleep(0.5)
