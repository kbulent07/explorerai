# YOLOX saf yardimcilarinin testleri. onnxruntime/model GEREKMEZ.
import numpy as np

from detection_backend import yolox_preproc, yolox_decode, nms, yolox_postprocess


def test_preproc_ratio_and_padding():
    img = np.full((100, 200, 3), 50, dtype=np.uint8)  # h=100, w=200
    chw, r = yolox_preproc(img, input_size=416)
    # ratio = min(416/100, 416/200) = 2.08
    assert abs(r - min(416 / 100, 416 / 200)) < 1e-6
    assert chw.shape == (3, 416, 416)
    assert chw.dtype == np.float32
    # padding bolgesi 114 olmali (sag alt kose)
    assert chw[0, -1, -1] == 114.0


def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, nms_thr=0.5)
    # ilk iki kutu cakisir -> biri elenir; ucuncu ayri kalir
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep


def test_postprocess_filters_class_and_threshold():
    # 2 aday: idx0 person(cls0) skor yuksek, idx1 person skor dusuk -> elenir
    # decoded: [cx, cy, w, h, obj, cls0, cls1]
    decoded = np.array([
        [50.0, 50.0, 20.0, 40.0, 0.9, 0.9, 0.1],   # score=0.81 -> kalir
        [80.0, 80.0, 10.0, 10.0, 0.5, 0.2, 0.1],   # score=0.10 -> elenir
    ])
    out = yolox_postprocess(decoded, ratio=1.0, conf_thr=0.3, nms_thr=0.45, class_id=0)
    assert len(out) == 1
    x, y, w, h, score = out[0]
    # cxcywh(50,50,20,40) -> xywh(40,30,20,40)
    assert (x, y, w, h) == (40, 30, 20, 40)
    assert score > 0.8


def test_postprocess_rescales_by_ratio():
    decoded = np.array([[100.0, 100.0, 40.0, 40.0, 1.0, 1.0]])  # tek sinif
    out = yolox_postprocess(decoded, ratio=2.0, conf_thr=0.1, nms_thr=0.45, class_id=0)
    x, y, w, h, _ = out[0]
    # xyxy(80,80,120,120)/2 = (40,40,60,60) -> xywh(40,40,20,20)
    assert (x, y, w, h) == (40, 40, 20, 20)


def test_decode_shapes():
    input_size = 64
    n = sum((input_size // s) ** 2 for s in (8, 16, 32))  # 64+16+4=84... (8->8^2=64,16->4^2=16,32->2^2=4)
    raw = np.zeros((n, 6), dtype=float)  # 1 sinif: 5+1
    raw[:, 4] = 0.0
    decoded = yolox_decode(raw, input_size=input_size)
    assert decoded.shape == (n, 6)
    # w,h = exp(0)*stride = stride; ilk satir stride=8 -> w=h=8
    assert abs(decoded[0, 2] - 8.0) < 1e-6
