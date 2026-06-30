# detection_backend.py
# -----------------------------------------------------------------------------
# Kisi (govde) tespiti arka uclari. Su an: YOLOX (ONNX, onnxruntime).
#
# YOLOX cikarimi SAF onnxruntime + numpy ile yazilir (torch / yolox pip paketi
# GEREKMEZ). Standart YOLOX ONNX export sozlesmesi: letterbox + 114 padding,
# NORMALIZASYON YOK; cikti grid-decode + NMS ister.
# -----------------------------------------------------------------------------

import logging
import os

import cv2 as cv
import numpy as np

log = logging.getLogger("facezoom.detection")


def yolox_preproc(img_bgr, input_size, pad_value=114):
    """BGR kare -> (CHW float32, ratio). Letterbox; normalizasyon yok (std YOLOX)."""
    ih, iw = img_bgr.shape[:2]
    padded = np.ones((input_size, input_size, 3), dtype=np.uint8) * pad_value
    r = min(input_size / ih, input_size / iw)
    nw, nh = max(1, int(iw * r)), max(1, int(ih * r))
    resized = cv.resize(img_bgr, (nw, nh), interpolation=cv.INTER_LINEAR)
    padded[:nh, :nw] = resized
    chw = np.ascontiguousarray(padded.transpose(2, 0, 1).astype(np.float32))
    return chw, r


def yolox_decode(outputs, input_size, strides=(8, 16, 32)):
    """Ham model ciktisini (N, 5+C) piksel cxcywh'e cozer (kare giris varsayar)."""
    grids = []
    expanded = []
    for stride in strides:
        g = input_size // stride
        xv, yv = np.meshgrid(np.arange(g), np.arange(g))
        grid = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(grid)
        expanded.append(np.full((grid.shape[0], 1), stride))
    grids = np.concatenate(grids, 0)
    expanded = np.concatenate(expanded, 0)
    out = outputs.astype(np.float32, copy=True)
    out[:, :2] = (out[:, :2] + grids) * expanded
    out[:, 2:4] = np.exp(out[:, 2:4]) * expanded
    return out


def nms(boxes_xyxy, scores, nms_thr):
    """Tek-sinif NMS. boxes: (M,4) xyxy. Tutulan indeksleri dondurur."""
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def yolox_postprocess(decoded, ratio, conf_thr, nms_thr, class_id=0):
    """decoded (N,5+C) piksel cxcywh -> [(x,y,w,h,score), ...] ORIJINAL uzayda."""
    boxes = decoded[:, :4]
    obj = decoded[:, 4]
    cls = decoded[:, 5:]
    scores = obj * cls[:, class_id]
    mask = scores >= conf_thr
    boxes, scores = boxes[mask], scores[mask]
    if boxes.shape[0] == 0:
        return []
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    xyxy /= ratio
    keep = nms(xyxy, scores, nms_thr)
    out = []
    for i in keep:
        x1, y1, x2, y2 = xyxy[i]
        out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(scores[i])))
    return out


# -------- Config ve YOLOX dedektor --------

_VALID_BACKENDS = ("mediapipe", "yolox_person")


def _as_float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def resolve_detection_config(config):
    """config.yaml dict'inden tespit ayarlarini guvenle cozer (varsayilan + dogrulama)."""
    backend = str(config.get("detector_backend", "mediapipe")).strip().lower()
    if backend not in _VALID_BACKENDS:
        log.warning("Gecersiz detector_backend '%s' -> mediapipe", backend)
        backend = "mediapipe"
    providers = config.get("yolox_providers") or ["CPUExecutionProvider"]
    if not isinstance(providers, (list, tuple)) or not all(isinstance(p, str) for p in providers):
        log.warning("Gecersiz yolox_providers -> ['CPUExecutionProvider']")
        providers = ["CPUExecutionProvider"]
    return {
        "backend": backend,
        "yolox_model": str(config.get("yolox_model", "models/yolox_nano.onnx")),
        "yolox_input_size": _as_int(config.get("yolox_input_size", 416), 416),
        "yolox_confidence": _as_float(config.get("yolox_confidence", 0.35), 0.35),
        "yolox_nms": _as_float(config.get("yolox_nms", 0.45), 0.45),
        "yolox_providers": list(providers),
        "person_min_size": _as_int(config.get("person_min_size", 40), 40),
        "bytetrack_track_activation_threshold":
            _as_float(config.get("bytetrack_track_activation_threshold", 0.25), 0.25),
        "bytetrack_lost_buffer": _as_int(config.get("bytetrack_lost_buffer", 30), 30),
        "bytetrack_min_matching_threshold":
            _as_float(config.get("bytetrack_min_matching_threshold", 0.8), 0.8),
    }


class PersonDetector:
    """Kisi tespiti arka ucu arayuzu."""
    def detect(self, frame_bgr):
        raise NotImplementedError


class YoloxPersonDetector(PersonDetector):
    """YOLOX ONNX kisi (COCO class 0) tespiti. onnxruntime tembel yuklenir."""

    PERSON_CLASS = 0

    def __init__(self, model_path, input_size=416, confidence=0.35, nms_thr=0.45,
                 providers=None, person_min_size=40):
        self.model_path = model_path
        self.input_size = int(input_size)
        self.confidence = float(confidence)
        self.nms_thr = float(nms_thr)
        self.providers = providers or ["CPUExecutionProvider"]
        self.person_min_size = int(person_min_size)
        self._session = None
        self._input_name = None

    def _ensure(self):
        if self._session is not None:
            return self._session
        import onnxruntime as ort
        log.info("YOLOX yukleniyor: %s (providers=%s)", self.model_path, self.providers)
        sess = ort.InferenceSession(self.model_path, providers=self.providers)
        active = sess.get_providers()
        if "CUDAExecutionProvider" in self.providers and "CUDAExecutionProvider" not in active:
            log.warning("CUDA provider istendi ama aktif degil; CPU kullaniliyor (%s)", active)
        input_name = sess.get_inputs()[0].name
        self._session = sess
        self._input_name = input_name
        return sess

    def detect(self, frame_bgr):
        sess = self._ensure()
        chw, ratio = yolox_preproc(frame_bgr, self.input_size)
        out = sess.run(None, {self._input_name: chw[None, :, :, :]})[0]
        decoded = yolox_decode(out[0], self.input_size)
        dets = yolox_postprocess(decoded, ratio, self.confidence, self.nms_thr,
                                 class_id=self.PERSON_CLASS)
        result = []
        for x, y, w, h, score in dets:
            if min(w, h) < self.person_min_size:
                continue
            result.append({"bbox": (x, y, w, h), "confidence": score})
        return result


def build_person_detector(resolved):
    """Cozulmus config'ten dedektor kur. Model yok/yuklenemiyorsa None dondur
    (worker mediapipe'e duser). ASLA istisna firlatmaz."""
    if resolved.get("backend") != "yolox_person":
        return None
    model = resolved.get("yolox_model")
    if not model or not os.path.exists(model):
        log.error("YOLOX modeli bulunamadi: %s -> mediapipe arka ucuna dusuluyor", model)
        return None
    try:
        return YoloxPersonDetector(
            model_path=model,
            input_size=resolved["yolox_input_size"],
            confidence=resolved["yolox_confidence"],
            nms_thr=resolved["yolox_nms"],
            providers=resolved["yolox_providers"],
            person_min_size=resolved["person_min_size"],
        )
    except Exception:
        log.exception("YOLOX dedektor kurulamadi -> mediapipe arka ucuna dusuluyor")
        return None
