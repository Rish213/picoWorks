# ui/vision_bridge.py
from PySide6.QtCore import QThread, Signal
import numpy as np
from core.picoVision import VisionEngine

class VisionQtBridge(QThread):
    # emits annotated BGR numpy array and timestamp (float)
    frame_ready = Signal(object, float)
    detection_ready = Signal(object, float)

    def __init__(self, source_url: str, onnx_path: str=None, conf=0.25):
        super().__init__()
        self.source_url = source_url
        self.running = False
        kwargs = {}
        if onnx_path:
            kwargs["onnx_path"] = onnx_path
            kwargs["use_onnx"] = True
        self.engine = VisionEngine(source=source_url, conf_thres=conf, **kwargs)

    def run(self):
        self.running = True

        def frame_cb(frame, ts):
            # emit as numpy BGR
            self.frame_ready.emit(frame, ts)

        def det_cb(dets, ts):
            self.detection_ready.emit(dets, ts)

        self.engine.on_frame = frame_cb
        self.engine.on_detection = det_cb
        self.engine.start()

        # keep thread alive until stopped
        while self.running and not self.engine._stop_event.is_set():
            self.msleep(50)

        # cleanup
        try:
            self.engine.stop()
        except:
            pass

    def stop(self):
        self.running = False
        try:
            self.engine.stop()
        except:
            pass
        self.wait()
