# detection_backend.py
# -----------------------------------------------------------------------------
# Kisi (govde) tespiti arka uclari. Su an: YOLOX (ONNX, onnxruntime).
#
# YOLOX cikarimi SAF onnxruntime + numpy ile yazilir (torch / yolox pip paketi
# GEREKMEZ). Standart YOLOX ONNX export sozlesmesi: letterbox + 114 padding,
# NORMALIZASYON YOK; cikti grid-decode + NMS ister.
# -----------------------------------------------------------------------------

import logging

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
