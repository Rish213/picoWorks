# core/picoVision.py
"""
picoVision - ONNX-first low-latency vision engine for DroneAI / picoWorks.
- Prefers ONNXRuntime (CUDA if available) for low-latency inference.
- Falls back to torch.hub model if ONNX not provided.
- Threaded capture / inference / output pipeline.
- UI-agnostic: use callbacks on_frame(frame, ts) and on_detection(dets, ts).
Notes:
 - ONNX parser assumes typical YOLO export: output (1, N, C) where C >= 6 (xywh + conf + classes) OR (1, M, 85).
 - If your ONNX uses a different signature, adjust _parse_onnx_output accordingly.
"""
import time
import threading
import queue
from typing import Callable, Optional, Tuple, List, Dict
import numpy as np
import cv2
import os

# Try to import onnxruntime and torch (torch optional fallback)
try:
    import onnxruntime as ort
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# config
DEFAULT_VIDEO_SRC = "rtsp://10.82.234.8:1945/"
MODEL_ONNX = "yolov11n.onnx"
MODEL_PT = "yolov11n.pt"
USE_ONNX = True
MAX_QUEUE = 4
INFERENCE_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
MAX_DETECTIONS = 100

# utils
def xywh2xyxy(box):
    x, y, w, h = box
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return [x1, y1, x2, y2]

def draw_boxes(frame: np.ndarray, detections: List[Dict], show_label=True) -> np.ndarray:
    for det in detections:
        x1,y1,x2,y2 = map(int, det["xyxy"])
        conf = det["conf"]
        label = det.get("label", str(det.get("class","")))
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        txt = f"{label} {conf:.2f}" if show_label else f"{conf:.2f}"
        cv2.putText(frame, txt, (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    return frame

# minimal NMS wrapper using cv2.dnn.NMSBoxes (expects [x,y,w,h])
def nms_boxes(boxes_xyxy, scores, iou_thresh):
    boxes_xywh = []
    for b in boxes_xyxy:
        x1,y1,x2,y2 = b
        boxes_xywh.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
    if len(boxes_xywh)==0:
        return []
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.0, nms_threshold=iou_thresh)
    # cv2 returns list of [i] or empty
    if len(indices)==0:
        return []
    inds = [int(i[0]) if isinstance(i, (list,tuple,np.ndarray)) else int(i) for i in indices]
    return inds

class VisionEngine:
    def __init__(self, source=None, conf_thres=0.25, iou_thres=0.45, input_size=640, **kwargs): # Added iou_thres, input_size
        self.source = source if source is not None else DEFAULT_VIDEO_SRC # Use default if None
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres # Added
        self.input_size = input_size # Added
        self.onnx_path = kwargs.get("onnx_path", None)
        self.ort_sess = None
        self.torch_model = None # Added for clarity if you plan fallback

        # --- Added Missing Initializations ---
        self._threads = []
        self._capture_q = queue.Queue(maxsize=MAX_QUEUE)
        self._result_q = queue.Queue(maxsize=MAX_QUEUE)
        self._stop_event = threading.Event()
        self.running = False
        # --- End Added ---

        # Callbacks
        self.on_frame = kwargs.get("on_frame", None)
        self.on_detection = kwargs.get("on_detection", None)

        # Check ONNX Runtime availability and providers
        if _HAS_ORT:
            self.available_providers = ort.get_available_providers()
        else:
             self.available_providers = []
             print("[VisionEngine] ⚠️ ONNX Runtime not found.")

        self.device = "CPU"
        self._load_model() # Call the loading function

    # --- REMOVED the outer _load_model definition ---

    # --- This is your INNER _load_model definition (keep it) ---
    def _load_model(self):
        # (Your existing _load_model code goes here - no changes needed inside it)
        # Make sure the indentation starts here
        if not self.onnx_path:
            default_path = os.path.join(os.path.dirname(__file__), "yolov11n.onnx")
            print(f"[VisionEngine] ⚠️ No model path provided — trying default: {default_path}")
            self.onnx_path = default_path

        if not os.path.exists(self.onnx_path):
            print(f"[VisionEngine] ❌ ONNX model not found at: {self.onnx_path}")
            self.ort_sess = None
            # --- Optional: Add PyTorch fallback here if desired ---
            # if _HAS_TORCH and not self.ort_sess:
            #     print("[VisionEngine] Attempting PyTorch fallback...")
            #     try:
            #         self.torch_model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # Example
            #         print("[VisionEngine] ✅ PyTorch model loaded.")
            #     except Exception as e_torch:
            #         print(f"[VisionEngine] ❌ PyTorch fallback failed: {e_torch}")
            # --- End Optional Fallback ---
            return

        print(f"[VisionEngine] Loading ONNX model: {self.onnx_path}")
        if not _HAS_ORT:
            print("[VisionEngine] ❌ Cannot load ONNX model, ONNX Runtime is not installed.")
            self.ort_sess = None
            return

        try:
            if "CUDAExecutionProvider" in self.available_providers:
                print("[VisionEngine] ✅ CUDA available — using GPU")
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.device = "GPU"
            else:
                print("[VisionEngine] ⚠️ CUDA not available — using CPU only")
                providers = ["CPUExecutionProvider"]

            self.ort_sess = ort.InferenceSession(self.onnx_path, providers=providers)
            
            if self.ort_sess is not None:
                try:
                    print("ORT inputs:", [(i.name, i.shape, i.type) for i in self.ort_sess.get_inputs()])
                except Exception as e_info:
                    print(f"[VisionEngine] ⚠️ Could not print input info: {e_info}")
            else:
                 print("[VisionEngine] ⚠️ ONNX session is None (model failed to load)")

        except Exception as e_load:
            print(f"[VisionEngine] ❌ Failed to load ONNX model: {e_load}")
            self.ort_sess = None
    
        


    def start(self):
        self._stop_event.clear()
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        t_cap = threading.Thread(target=self._capture_loop, daemon=True, name="cap-thread")
        t_inf = threading.Thread(target=self._inference_loop, daemon=True, name="inf-thread")
        t_out = threading.Thread(target=self._output_loop, daemon=True, name="out-thread")
        for t in (t_cap, t_inf, t_out):
            t.start()
            self._threads.append(t)
        print("[picoVision] pipeline started")

    def stop(self):
        self._stop_event.set()
        # release capture
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
        # wait threads to die
        for t in self._threads:
            if t.is_alive():
                try:
                    t.join(timeout=0.2)
                except:
                    pass
        self._threads = []
        print("[picoVision] pipeline stopped")

    def _capture_loop(self):
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            h,w = frame.shape[:2]
            if max(h,w) > 1280:
                scale = 1280 / max(h,w)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            ts = time.time()
            try:
                self._capture_q.put_nowait((frame, ts))
            except queue.Full:
                try:
                    self._capture_q.get_nowait()
                    self._capture_q.put_nowait((frame, ts))
                except:
                    pass

    def _preprocess_onnx(self, frame: np.ndarray):
        # Resize keeping aspect to square INFERENCE_SIZE with letterbox (like YOLO exports)
        img = frame.copy()
        h0, w0 = img.shape[:2]
        r = self.input_size / max(h0, w0)
        nh, nw = int(round(h0 * r)), int(round(w0 * r))
        img_resized = cv2.resize(img, (nw, nh))
        # pad to square
        pad_img = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_img[(self.input_size - nh)//2:(self.input_size - nh)//2+nh, (self.input_size - nw)//2:(self.input_size - nw)//2+nw] = img_resized
        # BGR->RGB, normalize
        img_in = pad_img[:, :, ::-1].astype(np.float32) / 255.0
        img_in = np.transpose(img_in, (2,0,1))[np.newaxis, ...]  # 1,C,H,W
        return img_in, (h0, w0), ((self.input_size - nw)//2, (self.input_size - nh)//2), (nw, nh)

    def _parse_onnx_output(self, outputs, orig_shape, pad, resized_wh):
        # Typical yolov5/ultralytics ONNX export returns one output with shape (1, N, 85)
        # where each row: [x, y, w, h, conf, c1, c2, ...]
        out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if isinstance(out, np.ndarray):
            arr = out
        else:
            arr = np.array(out)
        if arr.ndim == 3:
            arr = arr[0]  # (N, C)
        detections = []
        if arr.size == 0:
            return detections
        # if last dim >=6, try typical parse
        if arr.shape[1] >= 6:
            # xywh relative to padded image or absolute? Many exports are absolute pixel coords relative to input_size.
            # We'll assume coordinates are relative to input_size (x center, y center, w, h).
            for row in arr:
                x, y, w, h = row[0:4].tolist()
                conf = float(row[4])
                class_scores = row[5:]
                if class_scores.size == 0:
                    cls = 0
                else:
                    cls = int(np.argmax(class_scores))
                    conf = conf * float(class_scores[cls])
                if conf < self.conf_thres:
                    continue
                # convert xywh (input-space) to xyxy on original image using pad/rescale
                pad_x, pad_y = pad
                nw, nh = resized_wh
                # Convert center xy from input-space to original pixels
                x1 = (x - w/2 - pad_x) * (orig_shape[1] / nw)
                y1 = (y - h/2 - pad_y) * (orig_shape[0] / nh)
                x2 = (x + w/2 - pad_x) * (orig_shape[1] / nw)
                y2 = (y + h/2 - pad_y) * (orig_shape[0] / nh)
                detections.append({"xyxy":[x1,y1,x2,y2], "conf":conf, "class":int(cls), "label":str(int(cls))})
        else:
            # Unknown format; attempt to decode as (N,6) x1,y1,x2,y2,conf,class
            if arr.shape[1] == 6:
                for row in arr:
                    x1,y1,x2,y2,conf,cls = row.tolist()
                    if conf < self.conf_thres: continue
                    detections.append({"xyxy":[x1,y1,x2,y2], "conf":conf, "class":int(cls), "label":str(int(cls))})
        # run NMS
        if not detections:
            return []
        boxes = [d["xyxy"] for d in detections]
        scores = [float(d["conf"]) for d in detections]
        keep = nms_boxes(boxes, scores, self.iou_thres)
        kept = [detections[i] for i in keep][:MAX_DETECTIONS]
        return kept

    def _inference_loop(self):
        while not self._stop_event.is_set():
            try:
                frame, ts = self._capture_q.get(timeout=0.5)
            except queue.Empty:
                continue
            detections = []
            inf_start = time.time()
            try:
                if self.ort_sess is not None:
                    img_in, orig_shape, pad, resized_wh = self._preprocess_onnx(frame)
                    inp_name = self.ort_sess.get_inputs()[0].name
                    outputs = self.ort_sess.run(None, {inp_name: img_in})
                    detections = self._parse_onnx_output(outputs, orig_shape, pad, resized_wh)
                elif self.torch_model is not None:
                    # ultralytics style
                    preds = self.torch_model([frame], size=self.input_size)
                    det = preds.xyxy[0].cpu().numpy()
                    for row in det:
                        x1,y1,x2,y2,conf,cls = row.tolist()
                        if conf < self.conf_thres: continue
                        label = self.torch_model.names[int(cls)] if hasattr(self.torch_model, "names") else str(int(cls))
                        detections.append({"xyxy":[x1,y1,x2,y2], "conf":float(conf), "class":int(cls), "label":label})
                else:
                    detections = []
            except Exception as e:
                print(f"[picoVision] inference error: {e}")
                detections = []
            inf_time = time.time() - inf_start
            try:
                self._result_q.put_nowait((frame, ts, detections, inf_time))
            except queue.Full:
                try:
                    self._result_q.get_nowait()
                    self._result_q.put_nowait((frame, ts, detections, inf_time))
                except:
                    pass

    def _output_loop(self):
        last_display = time.time()
        fps_smooth = 0.0
        alpha = 0.2
        while not self._stop_event.is_set():
            try:
                frame, ts, detections, inf_time = self._result_q.get(timeout=0.5)
            except queue.Empty:
                continue
            now = time.time()
            fps = 1.0 / max(1e-6, (now - last_display))
            last_display = now
            fps_smooth = (1-alpha)*fps_smooth + alpha*fps

            out_frame = frame.copy()
            out_frame = draw_boxes(out_frame, detections)
            cv2.putText(out_frame, f"FPS: {fps_smooth:.1f} INF: {inf_time*1000:.0f}ms", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)

            if self.on_frame:
                try:
                    self.on_frame(out_frame, now)
                except Exception as e:
                    print(f"[picoVision] on_frame callback error: {e}")
            if self.on_detection:
                try:
                    self.on_detection(detections, now)
                except Exception as e:
                    print(f"[picoVision] on_detection callback error: {e}")
