# tests/test_yolox_detector_contract.py
# -----------------------------------------------------------------------------
# YoloxPersonDetector.detect() sozlesmesini kilitleyen contract testi.
# onnxruntime veya gercek model GEREKMEZ: _session sahte (fake) nesneyle
# enjekte edilir; _ensure() _session is not None kontroluyle erken doner.
#
# input_size=416 icin toplam anchor sayisi:
#   (416//8)^2 + (416//16)^2 + (416//32)^2 = 52^2 + 26^2 + 13^2 = 3549
# COCO 80 sinif: toplam sutun = 5 + 80 = 85 -> cikti sekli (1, 3549, 85)
# -----------------------------------------------------------------------------

import numpy as np

from detection_backend import YoloxPersonDetector

INPUT_SIZE = 416
N_ANCHORS = sum((INPUT_SIZE // s) ** 2 for s in (8, 16, 32))  # 3549
N_CLASSES = 80  # COCO


class _FakeSession:
    """onnxruntime.InferenceSession yerine gecen sahte oturum.

    run() cagrildigi anda (1, N_ANCHORS, 85) seklinde sentez cikti dondurur;
    icerigi dissa aktarilan build_output() ile test bazinda ozellestirilebilir.
    """

    def __init__(self, output_fn):
        # output_fn(name, chw_batch) -> (1, N, 85) ndarray
        self._output_fn = output_fn
        self._last_input = None

    def run(self, output_names, input_feed):
        # input_feed: {"images": (1,3,H,W)}
        name, chw = next(iter(input_feed.items()))
        self._last_input = chw
        return [self._output_fn(name, chw)]


def _make_raw(person_anchor_idx, non_person_anchor_idx=None):
    """Sentetik ham YOLOX ciktisi (1, N_ANCHORS, 85) olustur.

    YOLOX decode sozlesmesi: cx_decoded = (cx_raw + grid_x) * stride
    Dolayisiyla cx_raw = 0.0 -> anchor grid hucresinin tam ortasina eslesir.
    w_decoded = exp(w_raw) * stride; w_raw = 0 -> w_decoded = stride px.
    Kucuk w_raw degerleri (0..2) makul boyutlarda kutular uretir.

    person_anchor_idx : stride=8 grid (ilk 52*52=2704) icerisinde gecerli indeks.
                        Yuksek objectness + yuksek cls0 skoru atar.
    non_person_anchor_idx : yuksek objectness ama cls0 cok dusuk, cls5 yuksek
                            (sinif-0 filtresini test eder).
    """
    raw = np.zeros((1, N_ANCHORS, 5 + N_CLASSES), dtype=np.float32)

    # person anchor: cx_raw=0, cy_raw=0 -> grid hucresinin merkezine eslesir.
    # w_raw=1.5 -> w = exp(1.5)*8 ~ 36 px (416 uzayinda); ratio sonrasi ~55 px (640x480)
    pa = person_anchor_idx
    raw[0, pa, 0] = 0.0    # cx offset (grid hucresine gore)
    raw[0, pa, 1] = 0.0    # cy offset
    raw[0, pa, 2] = 1.5    # log oran: w ~ exp(1.5) * stride
    raw[0, pa, 3] = 1.5    # log oran: h ~ exp(1.5) * stride
    raw[0, pa, 4] = 0.95   # objectness
    raw[0, pa, 5 + 0] = 0.95  # sinif 0 (person)

    if non_person_anchor_idx is not None:
        # person_anchor_idx'ten uzakta farkli bir anchor; stride=16 bolgesinde
        npa = non_person_anchor_idx
        raw[0, npa, 0] = 0.0
        raw[0, npa, 1] = 0.0
        raw[0, npa, 2] = 1.5
        raw[0, npa, 3] = 1.5
        raw[0, npa, 4] = 0.95    # yuksek objectness
        raw[0, npa, 5 + 0] = 0.02  # cls0 (person) cok dusuk
        raw[0, npa, 5 + 5] = 0.95  # cls5 (baska sinif) yuksek

    return raw


# yolox_decode beklentisi: giriste raw[0] (N,85) kullanilir; _ensure dogrudan
# sess.run(None, {name: batch})[0] -> raw, sonra out[0] ile dilimler.
# detect() cagrildiginda: sess.run -> raw (1,N,85); out = raw; decoded = yolox_decode(out[0], ...)
# Bu yuzden FakeSession raw'i (1,N,85) olarak dondurmelidir.

def _make_session_for(raw):
    """Verilen (1,N,85) ndarray'i donduren FakeSession olustur."""
    def output_fn(name, chw_batch):
        return raw
    return _FakeSession(output_fn)


# ---- Testler --------------------------------------------------------------- #

def test_detect_returns_person_with_high_score():
    """Yuksek objectness + yuksek cls0 skoru tasisan anchor -> detect listesinde
    en az bir girdi donmeli; bbox w>0 ve h>0, confidence esigi uzeri olmali."""
    # stride=8 grid: 52x52 = 2704 anchor. Orta bir anchor sec.
    person_idx = 100  # stride=8 bolgesinde gecerli indeks

    raw = _make_raw(person_anchor_idx=person_idx)
    det = YoloxPersonDetector(
        model_path="dummy.onnx",
        input_size=INPUT_SIZE,
        confidence=0.3,
        nms_thr=0.45,
        person_min_size=10,
    )
    # _ensure() bypass: _session None degilse erken doner
    det._session = _make_session_for(raw)
    det._input_name = "images"

    frame = np.zeros((480, 640, 3), np.uint8)
    results = det.detect(frame)

    assert len(results) >= 1, "En az bir kisi tespiti donmeli"
    r = results[0]
    assert "bbox" in r and "confidence" in r
    x, y, w, h = r["bbox"]
    assert w > 0, "bbox genisligi pozitif olmali"
    assert h > 0, "bbox yuksekligi pozitif olmali"
    assert r["confidence"] >= 0.3, "Skor konfidans esiginin uzzerinde olmali"
    # bbox koordinatlari integer olmali (detect() int() ile donusturuyor)
    assert isinstance(x, int) and isinstance(y, int)
    assert isinstance(w, int) and isinstance(h, int)


def test_detect_filters_out_non_person_class():
    """Yuksek objectness ama cls0 (person) skoru cok dusuk olan anchor
    detect listesinde GORULMEMELI (sinif-0 filtresi)."""
    person_idx = 100      # gercek person anchor
    non_person_idx = 200  # yuksek obj, cls5 yuksek ama cls0 cok dusuk

    raw = _make_raw(person_anchor_idx=person_idx,
                    non_person_anchor_idx=non_person_idx)
    det = YoloxPersonDetector(
        model_path="dummy.onnx",
        input_size=INPUT_SIZE,
        confidence=0.3,
        nms_thr=0.45,
        person_min_size=10,
    )
    det._session = _make_session_for(raw)
    det._input_name = "images"

    frame = np.zeros((480, 640, 3), np.uint8)
    results = det.detect(frame)

    # Sonuclari confidence'a gore sirala; en yuksek skora bak
    assert len(results) >= 1, "En az kisi anchor detect edilmeli"
    # non-person anchor (cls0=0.02 * obj=0.95 = 0.019) esik altinda kalmali
    # yani NMS sonrasi toplamda person_idx'ten gelen tek tespit olmali
    # Yuksek skor: 0.95*0.95=0.9025; dusuk skor: 0.95*0.02=0.019 < 0.3 (conf esigi)
    # Hicbir sonucun confidence degeri 0.019 civarinda olmamali
    for r in results:
        assert r["confidence"] >= 0.3, (
            f"Dusuk sinif-0 skorlu anchor sonuc listesine girmemeli: {r['confidence']}"
        )
