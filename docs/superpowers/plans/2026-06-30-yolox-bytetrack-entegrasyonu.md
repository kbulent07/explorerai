# YOLOX Kişi Tespiti + supervision ByteTrack Entegrasyonu — Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FaceZoom'a, config ile seçilebilen bir "YOLOX kişi tespiti + supervision ByteTrack takibi" arka ucu eklemek; böylece kişi track kimliği örtüşme/uzaklık/baş dönüşünde kararlı kalır, yüz hâlâ best-shot + ArcFace için kullanılır.

**Architecture:** YOLOX (ONNX, onnxruntime) `det_img` üzerinde kişi (COCO class 0) tespit eder → supervision `ByteTrack` kararlı `track_id` verir → MediaPipe yüzleri içerme + IoU ile kişilere eşlenir → mevcut `compute_quality`/best-shot kişi `track_id` başına çalışır. Varsayılan arka uç `mediapipe`'tir; yeni yol yalnızca `detector_backend: yolox_person` ile devreye girer (opt-in, geri uyumlu).

**Tech Stack:** Python 3.12, onnxruntime (mevcut, CPU; GPU'ya provider ile hazır), supervision (ByteTrack), numpy, opencv, mediapipe (mevcut), insightface (mevcut).

## Global Constraints

- **Geri uyum:** `detector_backend` varsayılanı `mediapipe`; yeni anahtarlar yokken davranış birebir bugünküyle aynı kalmalı. Mevcut testler (`tests/test_tracking.py`, `tests/test_recent.py`, `tests/test_config_store.py`) her zaman yeşil kalmalı.
- **torch YOK:** YOLOX çıkarımı saf onnxruntime + numpy ile yazılır; `yolox` pip paketi veya `torch` eklenmez.
- **Sürüm zinciri kırılgan:** `numpy==2.5.0`, `mediapipe==0.10.35`, `insightface==1.0.1`, `onnxruntime==1.27.0` pinlidir ve birbirine duyarlıdır. `supervision` eklendikten sonra bu zincir bozulmamalı (Task 5'te doğrulanır).
- **CPU-önce:** Varsayılan provider `["CPUExecutionProvider"]`. GPU yalnızca config ile etkinleştirilebilir; kod GPU'yu zorunlu kılmaz.
- **Asla çökme:** Model/bağımlılık eksikse `yolox_person` sessizce `mediapipe`'e düşer ve loglar (canlı uygulama önceliği).
- **Türkçe yorum stili:** Mevcut kod ASCII-translit Türkçe yorum kullanır (ç/ş/ğ/ı/ö/ü yok). Yeni kod aynı stile uyar.
- **config.yaml gitignore'ludur:** Yeni anahtarlar `config.example.yaml`'a (versiyonlanır) eklenir; çalışan kopya için `config.yaml`'a da elle eklenir ama commit edilmez.
- **Test komutu:** venv aktifken `python -m pytest tests/ -v`.

---

## Task 0: Git deposunu başlat (opsiyonel — commit adımlarını etkinleştirir)

Proje şu an git deposu değil ama `.gitignore` mevcut ve doğru (config.yaml=sır, venv/captures/db hariç). Sonraki görevlerdeki commit adımlarının çalışması için git başlatılır. **Git istemiyorsanız bu görevi atlayın ve diğer görevlerdeki "Commit" adımlarını yok sayın.**

**Files:** (yok — yalnız VCS)

- [ ] **Step 1: Git başlat ve baseline durumu doğrula**

```bash
cd /d/proje/FaceZoom
git init
git status --short
```
Beklenen: `config.yaml`, `venv/`, `captures/`, `facezoom.db`, `.claude/`, `.agents/` listelenmez (gitignore çalışıyor).

- [ ] **Step 2: Baseline + tasarım/plan dokümanlarını commit'le**

```bash
git add -A
git commit -m "chore: baseline before yolox+bytetrack integration"
```
Beklenen: commit oluşur; `docs/superpowers/specs/...` ve `docs/superpowers/plans/...` dahil.

---

## Task 1: YOLOX ön/son-işleme saf fonksiyonları

YOLOX ONNX çıktısını çözen saf-numpy yardımcılar. onnxruntime veya model dosyası **gerektirmez** → tamamen birim test edilebilir.

**Files:**
- Create: `detection_backend.py`
- Test: `tests/test_yolox_postprocess.py`

**Interfaces:**
- Consumes: (yok)
- Produces:
  - `yolox_preproc(img_bgr, input_size, pad_value=114) -> (chw_float32: np.ndarray, ratio: float)`
  - `yolox_decode(outputs: np.ndarray, input_size: int, strides=(8,16,32)) -> np.ndarray` (girdi `(N, 5+C)`, çıktı aynı şekil; kutular piksel cxcywh)
  - `nms(boxes_xyxy: np.ndarray, scores: np.ndarray, nms_thr: float) -> list[int]`
  - `yolox_postprocess(decoded: np.ndarray, ratio: float, conf_thr: float, nms_thr: float, class_id=0) -> list[tuple]` (her eleman `(x, y, w, h, score)`, ORİJİNAL görüntü uzayında)

- [ ] **Step 1: Failing test yaz**

`tests/test_yolox_postprocess.py`:
```python
# YOLOX saf yardimcilarinin testleri. onnxruntime/model GEREKMEZ.
import numpy as np
import pytest

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
```

- [ ] **Step 2: Test'i çalıştır, başarısız olduğunu doğrula**

Run: `python -m pytest tests/test_yolox_postprocess.py -v`
Expected: FAIL — `ImportError: cannot import name 'yolox_preproc' from 'detection_backend'` (dosya yok).

- [ ] **Step 3: `detection_backend.py`'yi saf fonksiyonlarla oluştur**

```python
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
```

- [ ] **Step 4: Test'i çalıştır, geçtiğini doğrula**

Run: `python -m pytest tests/test_yolox_postprocess.py -v`
Expected: PASS (5 test).

- [ ] **Step 5: Commit**

```bash
git add detection_backend.py tests/test_yolox_postprocess.py
git commit -m "feat: YOLOX onnx pre/post-processing pure helpers"
```

---

## Task 2: Config çözümleme + `YoloxPersonDetector` sınıfı

Tespit config'ini doğrulayıp normalize eden saf fonksiyon ve YOLOX dedektör sınıfı (tembel onnxruntime oturumu).

**Files:**
- Modify: `detection_backend.py`
- Modify: `config.example.yaml` (yeni anahtarlar)
- Test: `tests/test_detection_config.py`

**Interfaces:**
- Consumes: `yolox_preproc`, `yolox_decode`, `yolox_postprocess` (Task 1)
- Produces:
  - `resolve_detection_config(config: dict) -> dict` — anahtarlar: `backend` (`"mediapipe"`|`"yolox_person"`), `yolox_model`, `yolox_input_size`, `yolox_confidence`, `yolox_nms`, `yolox_providers`, `person_min_size`, `bytetrack_track_activation_threshold`, `bytetrack_lost_buffer`, `bytetrack_min_matching_threshold`
  - `class PersonDetector` — `detect(frame_bgr) -> list[dict]`, her eleman `{"bbox": (x,y,w,h), "confidence": float}`
  - `class YoloxPersonDetector(PersonDetector)` — `__init__(model_path, input_size, confidence, nms_thr, providers, person_min_size)`; tembel yükler
  - `build_person_detector(resolved: dict) -> PersonDetector | None` — model yoksa/yüklenemezse `None` döndürür (worker fallback yapar)

- [ ] **Step 1: Failing test yaz**

`tests/test_detection_config.py`:
```python
# resolve_detection_config + build_person_detector (model yok) testleri.
import pytest

from detection_backend import resolve_detection_config, build_person_detector


def test_defaults_when_empty():
    r = resolve_detection_config({})
    assert r["backend"] == "mediapipe"
    assert r["yolox_input_size"] == 416
    assert r["yolox_providers"] == ["CPUExecutionProvider"]
    assert 0.0 < r["yolox_confidence"] < 1.0


def test_invalid_backend_falls_back_to_mediapipe():
    r = resolve_detection_config({"detector_backend": "uydurma"})
    assert r["backend"] == "mediapipe"


def test_overrides_are_read():
    r = resolve_detection_config({
        "detector_backend": "yolox_person",
        "yolox_input_size": 640,
        "yolox_confidence": 0.5,
        "yolox_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "bytetrack_lost_buffer": 50,
    })
    assert r["backend"] == "yolox_person"
    assert r["yolox_input_size"] == 640
    assert r["yolox_confidence"] == 0.5
    assert r["yolox_providers"][0] == "CUDAExecutionProvider"
    assert r["bytetrack_lost_buffer"] == 50


def test_build_detector_returns_none_when_model_missing():
    r = resolve_detection_config({
        "detector_backend": "yolox_person",
        "yolox_model": "models/_yok_olan_model_.onnx",
    })
    # Model dosyasi yok -> None (worker mediapipe'e duser), istisna FIRLATMAZ
    assert build_person_detector(r) is None
```

- [ ] **Step 2: Test'i çalıştır, başarısız olduğunu doğrula**

Run: `python -m pytest tests/test_detection_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_detection_config'`.

- [ ] **Step 3: `detection_backend.py`'ye config + dedektör ekle**

`detection_backend.py` sonuna ekle:
```python
import os

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
        self._session = sess
        self._input_name = sess.get_inputs()[0].name
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
```

- [ ] **Step 4: `config.example.yaml`'a yeni anahtarları ekle**

`config.example.yaml` içinde `# ---- algilama / takip ----` bölümünün ALTINA ekle:
```yaml
# ---- tespit arka ucu (YOLOX kisi tespiti + ByteTrack) ----
detector_backend: mediapipe        # varsayilan (geri uyumlu) | yolox_person
yolox_model: "models/yolox_nano.onnx"   # bkz. README: model edinme
yolox_input_size: 416              # nano/tiny varsayilani
yolox_confidence: 0.35             # kisi skoru esigi
yolox_nms: 0.45                    # NMS IoU esigi
yolox_providers: ["CPUExecutionProvider"]   # GPU: ["CUDAExecutionProvider","CPUExecutionProvider"]
person_min_size: 40                # px (detect uzayi); kucuk kisileri ele
bytetrack_track_activation_threshold: 0.25
bytetrack_lost_buffer: 30          # kac kare kayip track tutulsun
bytetrack_min_matching_threshold: 0.8
```

- [ ] **Step 5: Test'i çalıştır, geçtiğini doğrula**

Run: `python -m pytest tests/test_detection_config.py -v`
Expected: PASS (4 test).

- [ ] **Step 6: Commit**

```bash
git add detection_backend.py tests/test_detection_config.py config.example.yaml
git commit -m "feat: detection config resolver + YoloxPersonDetector"
```

---

## Task 3: Best-shot ortak mantığını taban sınıfa çıkar (refactor)

`FaceTrackerManager`'ın best-shot/finalize mantığı (`record_quality`, `collect_finished`, `flush_all`) `PersonTrackManager` ile paylaşılacak. DRY için taban sınıfa taşınır. Davranış değişmez; mevcut testler güvenlik ağıdır.

**Files:**
- Modify: `tracking.py`
- Test: `tests/test_tracking.py` (mevcut — değişmeden geçmeli)

**Interfaces:**
- Consumes: mevcut `Track` sınıfı
- Produces:
  - `class BestShotTrackManager` — alanlar `tracks: dict`, `track_timeout: float`; metodlar `record_quality(track_id, score, hires_crop, hires_bbox, now)`, `collect_finished(now) -> list[Track]`, `flush_all(now) -> list[Track]`
  - `FaceTrackerManager(BestShotTrackManager)` — `update`, `__init__` aynı public davranış

- [ ] **Step 1: Mevcut testlerin yeşil olduğunu doğrula (refactor öncesi taban çizgisi)**

Run: `python -m pytest tests/test_tracking.py -v`
Expected: PASS (mevcut testler).

- [ ] **Step 2: `tracking.py`'de taban sınıfı oluştur ve `FaceTrackerManager`'ı miras aldır**

`tracking.py` içinde `class FaceTrackerManager:` tanımını şununla değiştir (mevcut `record_quality`, `collect_finished`, `flush_all` gövdeleri taban sınıfa **birebir** taşınır; `update` ve `__init__` FaceTrackerManager'da kalır):
```python
class BestShotTrackManager:
    """Best-shot tutma + zaman asimiyla finalize ortak mantigi. Alt siniflar
    `update()` ile track_id atar; bu taban best-shot kaydi/finalize'i saglar."""

    def __init__(self, track_timeout=2.0):
        self.track_timeout = track_timeout
        self.tracks = {}            # track_id -> Track
        self._next_id = 1

    def record_quality(self, track_id, score, hires_crop, hires_bbox, now):
        tr = self.tracks.get(track_id)
        if tr is not None:
            tr.maybe_update_best(score, hires_crop, hires_bbox, now)

    def collect_finished(self, now):
        """track_timeout'u asan track'leri dondur ve listeden cikar."""
        finished = []
        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            if (now - tr.last_seen) > self.track_timeout:
                del self.tracks[tid]
                if tr.best_crop is not None and tr.best_score > 0:
                    finished.append(tr)
                else:
                    log.debug("Track %s kayda deger kare olmadan bitti", tid)
        return finished

    def flush_all(self, now):
        """Kapanissta tum aktif track'leri finalize et."""
        finished = []
        for tr in self.tracks.values():
            if tr.best_crop is not None and tr.best_score > 0:
                finished.append(tr)
        self.tracks.clear()
        return finished


class FaceTrackerManager(BestShotTrackManager):
    """Detect bbox'lari kareler arasi eslestirip track_id atar; best-shot tutar.

    track_timeout boyunca guncellenmeyen track'ler "biter" ve finalize edilir.
    """

    def __init__(self, iou_threshold=0.3, max_center_dist=120.0, track_timeout=2.0):
        super().__init__(track_timeout=track_timeout)
        self.iou_threshold = iou_threshold
        self.max_center_dist = max_center_dist

    def update(self, detect_bboxes, now):
        # ... MEVCUT update() govdesi AYNEN korunur (degistirme) ...
```

> Not: `update()` gövdesi mevcut haliyle kalır. `record_quality`/`collect_finished`/`flush_all` artık FaceTrackerManager'dan SİLİNİR (taban sınıftan miras alınır). `_next_id` taban sınıfa taşındı.

- [ ] **Step 3: Mevcut testleri çalıştır, hâlâ geçtiğini doğrula**

Run: `python -m pytest tests/test_tracking.py -v`
Expected: PASS (davranış değişmedi).

- [ ] **Step 4: Commit**

```bash
git add tracking.py
git commit -m "refactor: extract BestShotTrackManager base from FaceTrackerManager"
```

---

## Task 4: `associate_faces_to_persons` saf fonksiyonu

Yüzleri (MediaPipe) kişilere (ByteTrack) eşleyen saf fonksiyon: merkez-içerme, çoklu adayda IoU tie-break (eşitlikte en küçük track_id), içeren-kişisiz yüz düşer.

**Files:**
- Modify: `tracking.py`
- Test: `tests/test_face_person_association.py`

**Interfaces:**
- Consumes: mevcut `_iou(a, b)` (tracking.py)
- Produces: `associate_faces_to_persons(faces: list[dict], person_tracks: list[tuple]) -> list[tuple]` — `faces` elemanı `{"bbox":(x,y,w,h), ...}`; `person_tracks` elemanı `(track_id:int, bbox:(x,y,w,h))`; dönüş `[(track_id, face_dict), ...]`

- [ ] **Step 1: Failing test yaz**

`tests/test_face_person_association.py`:
```python
from tracking import associate_faces_to_persons


def _face(x, y, w, h):
    return {"bbox": (x, y, w, h), "confidence": 0.9}


def test_face_inside_single_person():
    faces = [_face(50, 30, 20, 20)]        # merkez (60,40)
    persons = [(7, (40, 20, 60, 120))]     # (40..100, 20..140) icerir
    out = associate_faces_to_persons(faces, persons)
    assert out == [(7, faces[0])]


def test_face_without_container_is_dropped():
    faces = [_face(500, 500, 20, 20)]
    persons = [(1, (0, 0, 100, 100))]
    assert associate_faces_to_persons(faces, persons) == []


def test_multi_candidate_picks_highest_iou():
    # Yuz merkezi iki kisi kutusunda da; daha cok ortusen kazanir
    face = _face(45, 45, 20, 20)           # merkez (55,55)
    p_small = (2, (50, 50, 60, 60))        # yuzle az ortusur
    p_big = (1, (0, 0, 120, 120))          # yuzu tamamen icerir -> daha yuksek IoU
    out = associate_faces_to_persons([face], [p_small, p_big])
    assert out == [(1, face)]


def test_tie_break_smallest_track_id():
    # Iki ozdes kisi kutusu (esit IoU) -> en kucuk track_id secilir
    face = _face(45, 45, 20, 20)
    out = associate_faces_to_persons([face], [(5, (0, 0, 120, 120)), (3, (0, 0, 120, 120))])
    assert out == [(3, face)]
```

- [ ] **Step 2: Test'i çalıştır, başarısız olduğunu doğrula**

Run: `python -m pytest tests/test_face_person_association.py -v`
Expected: FAIL — `ImportError: cannot import name 'associate_faces_to_persons'`.

- [ ] **Step 3: `tracking.py`'ye fonksiyonu ekle**

`tracking.py` içinde `_center` yardımcısının ALTINA ekle:
```python
def associate_faces_to_persons(faces, person_tracks):
    """Her yuzu, merkezini iceren kisi kutusuna esle. Coklu adayda en yuksek
    IoU'lu kisi (esitlikte en kucuk track_id). Iceren kisisi olmayan yuz dusER.

    faces        : [{"bbox": (x,y,w,h), ...}, ...]
    person_tracks: [(track_id, (x,y,w,h)), ...]
    donus        : [(track_id, face_dict), ...]
    """
    assignments = []
    for face in faces:
        fx, fy, fw, fh = face["bbox"]
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        candidates = []
        for tid, pbox in person_tracks:
            px, py, pw, ph = pbox
            if px <= cx <= px + pw and py <= cy <= py + ph:
                candidates.append((tid, pbox))
        if not candidates:
            continue
        # En yuksek IoU; esitlikte en kucuk track_id (-tid ile max'ta belirleyici)
        best_tid, _ = max(
            candidates,
            key=lambda c: (_iou(face["bbox"], c[1]), -c[0]),
        )
        assignments.append((best_tid, face))
    return assignments
```

- [ ] **Step 4: Test'i çalıştır, geçtiğini doğrula**

Run: `python -m pytest tests/test_face_person_association.py -v`
Expected: PASS (4 test).

- [ ] **Step 5: Commit**

```bash
git add tracking.py tests/test_face_person_association.py
git commit -m "feat: associate_faces_to_persons (containment + IoU tie-break)"
```

---

## Task 5: `PersonTrackManager` (supervision ByteTrack sarmalayıcı)

`supervision` bağımlılığını ekler ve ByteTrack'i `BestShotTrackManager` arayüzüne sarar.

**Files:**
- Modify: `requirements.txt`
- Modify: `tracking.py`
- Test: `tests/test_person_track_manager.py`

**Interfaces:**
- Consumes: `BestShotTrackManager` (Task 3), `Track` (tracking.py), `supervision.ByteTrack`
- Produces:
  - `class PersonTrackManager(BestShotTrackManager)` — `__init__(track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=12, track_timeout=2.0)`; `update(detections: list[dict], now: float) -> list[tuple]` — `detections` elemanı `{"bbox":(x,y,w,h), "confidence":float}`; dönüş `[(track_id, bbox), ...]`

- [ ] **Step 1: `supervision`'ı kur ve sürüm zincirini doğrula (KRİTİK)**

```bash
python -m pip install supervision
python -c "import supervision; print('sv', supervision.__version__)"
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import torch" 2>&1 || echo "OK: torch yok (beklenen)"
python -c "import mediapipe, insightface, onnxruntime, cv2; print('zincir OK')"
python -m pytest tests/ -v
```
Beklenen: supervision import olur, **torch kurulMAZ**, `numpy 2.5.0` korunur, mediapipe/insightface/onnxruntime import zinciri ve **tüm mevcut testler** yeşil kalır.

> **Kırılırsa (numpy oynadıysa / mediapipe import patladıysa):** supervision'ı kaldır (`pip uninstall -y supervision`), numpy 2.5 ile uyumlu bir supervision sürümü dene (`pip install "supervision==0.25.*"`), tekrar doğrula. Hâlâ çakışırsa `pip install --no-deps supervision` + eksik transitive bağımlılığı (`scipy`) elle ekle. Çözülene kadar Step 2'ye geçme. Çalışan sürümü Step 5'te `requirements.txt`'e pinle.

- [ ] **Step 2: Failing test yaz**

`tests/test_person_track_manager.py`:
```python
# PersonTrackManager testleri. supervision kuruluysa calisir, degilse atlanir.
import pytest

sv = pytest.importorskip("supervision")

import numpy as np
from tracking import PersonTrackManager


def _det(x, y, w, h, conf=0.9):
    return {"bbox": (x, y, w, h), "confidence": conf}


def test_stable_track_id_across_frames():
    m = PersonTrackManager(track_activation_threshold=0.1, frame_rate=10,
                           track_timeout=5.0)
    # Ayni kisi hafifce hareket eder -> ID degismemeli
    ids = []
    for dx in range(0, 30, 3):
        out = m.update([_det(100 + dx, 100, 50, 120)], now=float(dx))
        if out:
            ids.append(out[0][0])
    assert len(set(ids)) == 1, f"track_id kararsiz: {ids}"


def test_empty_update_no_crash():
    m = PersonTrackManager()
    assert m.update([], now=0.0) == []


def test_collect_finished_after_timeout():
    m = PersonTrackManager(track_activation_threshold=0.1, frame_rate=10,
                           track_timeout=1.0)
    out = m.update([_det(100, 100, 50, 120)], now=0.0)
    assert out, "track olusmali"
    tid = out[0][0]
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    m.record_quality(tid, 0.5, crop, (100, 100, 50, 120), now=0.0)
    finished = m.collect_finished(now=5.0)  # timeout asildi
    assert any(tr.track_id == tid for tr in finished)
```

- [ ] **Step 3: Test'i çalıştır, başarısız olduğunu doğrula**

Run: `python -m pytest tests/test_person_track_manager.py -v`
Expected: FAIL — `ImportError: cannot import name 'PersonTrackManager'`.

- [ ] **Step 4: `tracking.py`'ye `PersonTrackManager` ekle**

`tracking.py` içinde `FaceTrackerManager` sınıfının ALTINA ekle:
```python
class PersonTrackManager(BestShotTrackManager):
    """supervision ByteTrack sarmalayici. Kisi tespitlerinden kararli track_id
    uretir; best-shot/finalize taban siniftan gelir.

    update() FaceTrackerManager'dan FARKLI olarak guven (confidence) iceren
    tespit sozlukleri alir (ByteTrack yuksek/dussuk skor eslesmesi icin gerekir).
    """

    def __init__(self, track_activation_threshold=0.25, lost_track_buffer=30,
                 minimum_matching_threshold=0.8, frame_rate=12, track_timeout=2.0):
        super().__init__(track_timeout=track_timeout)
        import supervision as sv
        self._sv = sv
        self.byte_track = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections, now):
        """detections: [{"bbox":(x,y,w,h), "confidence":float}, ...].
        Donus: [(track_id, (x,y,w,h)), ...] (bu karede track_id alanlar)."""
        sv = self._sv
        if not detections:
            sv_det = sv.Detections.empty()
        else:
            xyxy = np.array(
                [[x, y, x + w, y + h] for (x, y, w, h) in (d["bbox"] for d in detections)],
                dtype=float,
            )
            conf = np.array([float(d["confidence"]) for d in detections], dtype=float)
            sv_det = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros(len(detections), dtype=int),
            )
        tracked = self.byte_track.update_with_detections(sv_det)

        assignments = []
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = tracked.xyxy[i]
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            tr = self.tracks.get(tid)
            if tr is None:
                self.tracks[tid] = Track(tid, bbox, now)
            else:
                tr.bbox = bbox
                tr.last_seen = now
            assignments.append((tid, bbox))
        return assignments
```

- [ ] **Step 5: Test'i çalıştır + `requirements.txt`'i güncelle**

Run: `python -m pytest tests/test_person_track_manager.py -v`
Expected: PASS (3 test).

`requirements.txt`'e ekle (Step 1'de doğrulanan tam sürümü kullan; örnek):
```
# Kisi takibi (ByteTrack) - torch'suz, numpy/scipy tabanli
supervision==0.25.1
```

- [ ] **Step 6: Tüm test paketini çalıştır (regresyon)**

Run: `python -m pytest tests/ -v`
Expected: PASS (tüm testler).

- [ ] **Step 7: Commit**

```bash
git add tracking.py tests/test_person_track_manager.py requirements.txt
git commit -m "feat: PersonTrackManager wrapping supervision ByteTrack"
```

---

## Task 6: `yolox_person` arka ucunu `worker.py`'ye bağla

`CameraWorker`'ı `detector_backend`'e göre dallandır: yeni yolda YOLOX kişi tespiti → ByteTrack → yüz↔kişi eşleme → kişi başına best-shot. Varsayılan `mediapipe` yolu **değişmez**.

**Files:**
- Modify: `worker.py`
- Modify: `config.yaml` (yerel çalışan kopya — commit edilmez)

**Interfaces:**
- Consumes: `resolve_detection_config`, `build_person_detector` (Task 2), `PersonTrackManager`, `associate_faces_to_persons` (Task 4/5), mevcut `FaceTracker`, `compute_quality`, `FaceTrackerManager`
- Produces: (worker içsel davranış; dış imza değişmez)

- [ ] **Step 1: `worker.py` importlarını ve `__init__` dallanmasını güncelle**

`worker.py` üst importlarını değiştir:
```python
from framing import FaceTracker, FrameTransformer
from tracking import (FaceTrackerManager, PersonTrackManager,
                      associate_faces_to_persons, compute_quality)
from detection_backend import resolve_detection_config, build_person_detector
```

`__init__` içinde `self.manager = FaceTrackerManager(...)` bloğunu (mevcut satır ~71-74) şununla değiştir:
```python
        # Tespit arka ucu: mediapipe (varsayilan) | yolox_person
        self._det_cfg = resolve_detection_config(config)
        self._person_detector = build_person_detector(self._det_cfg)
        # build_person_detector model yoksa None doner -> guvenli mediapipe fallback
        if self._det_cfg["backend"] == "yolox_person" and self._person_detector is not None:
            self.backend = "yolox_person"
            self.manager = PersonTrackManager(
                track_activation_threshold=self._det_cfg["bytetrack_track_activation_threshold"],
                lost_track_buffer=self._det_cfg["bytetrack_lost_buffer"],
                minimum_matching_threshold=self._det_cfg["bytetrack_min_matching_threshold"],
                frame_rate=max(1, int(config.get("preview_fps", 12))),
                track_timeout=config.get("track_timeout", 2.0),
            )
            log.info("Tespit arka ucu: yolox_person (ByteTrack)")
        else:
            if self._det_cfg["backend"] == "yolox_person":
                log.warning("yolox_person istendi ama dedektor kurulamadi -> mediapipe")
            self.backend = "mediapipe"
            self.manager = FaceTrackerManager(
                iou_threshold=config.get("track_iou_threshold", 0.3),
                track_timeout=config.get("track_timeout", 2.0),
            )
```

> `self.tracker` (MediaPipe `FaceTracker`) her iki arka uçta da kurulu kalır — yüz tespiti ikisinde de gerekir. `max_center_dist` güncellemesi (process içindeki `self.manager.max_center_dist = ...` satırı) yalnızca FaceTrackerManager'da anlamlı; Step 2'de korunur (PersonTrackManager'da o özniteliğe yazmak zararsızdır ama Step 2'de koşula alınır).

- [ ] **Step 2: `process()` detect bloğunu dallandır**

`worker.py` `process()` içinde `self.manager.max_center_dist = ...` satırını koşula al:
```python
        # Merkez-mesafe esigi yalniz mediapipe (FaceTrackerManager) yolunda anlamli
        if self.backend == "mediapipe":
            self.manager.max_center_dist = max(1.0, self._center_dist_factor * dw)
```

`if run_detect:` bloğunun içini arka uca göre ayır. Mevcut blok (satır ~147-177) şununla değiştirilir:
```python
        if run_detect:
            self._last_detect_id = det_id
            det_img = cv.resize(det_source, (dw, dh)) if det_resize else det_source

            # Yuz tespiti her iki arka uçta da gerekli (best-shot + frontallik)
            faces = self.tracker.detect(det_img)
            faces = [f for f in faces if f["bbox"][3] >= self.min_face_size]
            self._faces = faces
            frame_area = float(hw * hh)

            if self.backend == "yolox_person":
                # 1) YOLOX kisi tespiti -> 2) ByteTrack track_id
                persons = self._person_detector.detect(det_img)
                person_tracks = self.manager.update(persons, now)
                # 3) yuzleri iceren kisiye esle -> kisi track_id devralir
                pairs = associate_faces_to_persons(faces, person_tracks)
                dropped = len(faces) - len(pairs)
                if dropped > 0:
                    log.debug("%d yuz iceren kisi kutusu bulunamadi (dusuruldu)", dropped)
                for tid, face in pairs:
                    self._record_best(tid, face, sx, sy, hires_frame, frame_area, now)
            else:
                # mediapipe: yuz bbox'lari dogrudan track edilir (mevcut davranis)
                detect_bboxes = [f["bbox"] for f in faces]
                assignments = self.manager.update(detect_bboxes, now)
                bbox_to_face = {tuple(f["bbox"]): f for f in faces}
                for tid, dbbox in assignments:
                    face = bbox_to_face.get(tuple(dbbox))
                    if face is None:
                        continue
                    self._record_best(tid, face, sx, sy, hires_frame, frame_area, now)
```

Ortak best-shot kaydı, tekrar etmemek için yeni bir yardımcı metoda taşınır. `process()`'in ÜSTÜNE (sınıf metodu olarak) ekle:
```python
    def _record_best(self, tid, face, sx, sy, hires_frame, frame_area, now):
        """Bir yuz icin hires kirpinti + kalite skorunu hesaplayip manager'a yaz."""
        hbbox = scale_bbox(face["bbox"], sx, sy)
        crop = crop_with_margin(hires_frame, hbbox, self.crop_margin)
        if crop is None:
            return
        hface = {
            "bbox": hbbox,
            "confidence": face["confidence"],
            "keypoints": {k: (int(v[0] * sx), int(v[1] * sy))
                          for k, v in face.get("keypoints", {}).items()},
        }
        score = compute_quality(hface, crop, frame_area, self.weights)
        self.manager.record_quality(tid, score, crop, hbbox, now)
```

> Not: `collect_finished` / `_emit_capture` ve canlı zoom bloğu (en büyük yüz hedefi) DEĞİŞMEZ — `self._faces` her iki yolda da doldurulduğu için zoom aynen çalışır.

- [ ] **Step 3: Mevcut testlerin hâlâ geçtiğini doğrula (mediapipe yolu bozulmadı)**

Run: `python -m pytest tests/ -v`
Expected: PASS (worker testi yok; tracking/recent/config/detection testleri yeşil — mediapipe yolu mantığı değişmedi).

- [ ] **Step 4: Manuel uçtan uca doğrulama (mediapipe — regresyon)**

`config.yaml`'da `detector_backend: mediapipe` (veya anahtar yok) iken uygulamayı başlat:
```bash
python main.py
```
Beklenen: Bugünkü davranış — yüzler tespit edilir, zoom + best-shot çalışır. Hata/uyarı yok.

- [ ] **Step 5: Manuel uçtan uca doğrulama (yolox_person)**

Önce YOLOX modelini edin (Task 7 README adımı). Sonra `config.yaml`'a ekle:
```yaml
detector_backend: yolox_person
yolox_model: "models/yolox_nano.onnx"
```
```bash
python main.py
```
Beklenen: Log "Tespit arka ucu: yolox_person (ByteTrack)". Kişiler tespit edilir; yüz görünmediğinde bile kişi takip kimliği sürer; yüz görününce best-shot/yakalama oluşur. Model yoksa otomatik "mediapipe" fallback + uyarı.

- [ ] **Step 6: Commit**

```bash
git add worker.py
git commit -m "feat: wire yolox_person backend into CameraWorker"
```

---

## Task 7: YOLOX modeli edinme — README + DOCKER dokümantasyonu

`yolox_person` arka ucunu çalıştırmak için gereken model dosyasının nasıl edinileceğini belgele.

**Files:**
- Modify: `README.md`
- Modify: `DOCKER.md`

- [ ] **Step 1: `README.md`'ye YOLOX modeli bölümü ekle**

MediaPipe model indirme bölümünün yakınına ekle:
```markdown
### (Opsiyonel) YOLOX kişi tespiti + ByteTrack

`detector_backend: yolox_person` ile çalışmak için bir YOLOX ONNX modeli gerekir
(varsayılan yol: `models/yolox_nano.onnx`). Edinme yolları:

1. **Hazır ONNX:** YOLOX GitHub sürümler (releases) sayfasındaki ONNXRuntime demo
   varlıklarından `yolox_nano.onnx` indirin.
2. **Kendiniz export:** YOLOX deposunu klonlayıp (torch gerekir, tek seferlik, çevrimdışı):
   `python tools/export_onnx.py --output-name yolox_nano.onnx -n yolox-nano -c yolox_nano.pth`

İndirilen dosyayı `models/yolox_nano.onnx` olarak yerleştirin. Model dosyası git'e
**konmaz** (büyük binary). Daha hızlı/az isabetli alternatif: `yolox_tiny.onnx`
(config'te `yolox_model` + `yolox_input_size` güncelleyin).

GPU (NVIDIA, isteğe bağlı): `pip install onnxruntime-gpu` + CUDA/cuDNN kurun ve
config'te `yolox_providers: ["CUDAExecutionProvider","CPUExecutionProvider"]` yapın.
```

- [ ] **Step 2: `DOCKER.md`'ye not ekle**

```markdown
### YOLOX modeli (yolox_person arka ucu için)

`detector_backend: yolox_person` kullanacaksanız `models/yolox_nano.onnx` dosyasını
imaja dahil edin veya bir volume ile bağlayın. `models/` klasörü `.dockerignore`'da
hariç tutulmuşsa, modeli runtime'da volume olarak mount edin:
`-v $(pwd)/models:/app/models`. CPU imajında `onnxruntime` (CPU) yeterlidir.
```

- [ ] **Step 3: Commit**

```bash
git add README.md DOCKER.md
git commit -m "docs: YOLOX model acquisition for yolox_person backend"
```

---

## Self-Review (plan yazarı tarafından yapıldı)

**1. Spec coverage:**
- §3.1 YoloxPersonDetector → Task 1+2 ✓
- §3.2 PersonTrackManager + associate_faces_to_persons → Task 3+4+5 ✓
- §3.3 worker dallanma → Task 6 ✓
- §3.4 recognition/recent değişmez → hiçbir task dokunmuyor ✓
- §4 veri akışı → Task 6 Step 2 ✓
- §5 config anahtarları → Task 2 Step 4 (example) + Task 6 Step 5 (yerel). **Sapma:** spec "config_store.py doğrulaması" demişti; codebase config_store.py yalnız kamera CRUD yapıyor, skalerler `config.get` ile okunuyor → doğrulama mevcut desene uygun olarak `detection_backend.resolve_detection_config`'e konuldu (Task 2). config_store.py'ye dokunulmadı. ✓ (kullanıcıya bildirildi)
- §6 bağımlılıklar (supervision, onnxruntime mevcut, model repoya konmaz) → Task 5 + Task 7 ✓
- §7 hata yönetimi/fallback → Task 2 build_person_detector + Task 6 __init__ fallback ✓
- §8 açık konular (zoom hedefi=en büyük yüz korundu; performans manuel ölçüm) → Task 6 notu + Step 5 ✓
- §9 testler (3 yeni dosya + regresyon) → Task 1/2/4/5 ✓
- §10 geri uyum (varsayılan mediapipe) → Global Constraints + Task 6 ✓
- §11 etkilenen dosyalar → tüm task'lara dağıtıldı ✓ (framing.py'ye dokunulmadı, doğru)

**2. Placeholder taraması:** "TBD/TODO/sonra" yok. requirements.txt supervision sürümü Task 5 Step 1'de doğrulanan değere pinlenir (örnek 0.25.1 verildi, doğrulamaya bağlı).

**3. Tip tutarlılığı:** `detect()` → `[{"bbox":(x,y,w,h),"confidence":float}]` (Task 2) ↔ `PersonTrackManager.update(detections)` aynı şekli tüketir (Task 5) ↔ worker `persons` olarak geçirir (Task 6) ✓. `associate_faces_to_persons(faces, person_tracks)` `[(tid, (x,y,w,h))]` alır ↔ `manager.update` `[(tid, bbox)]` döndürür ✓. `BestShotTrackManager` metod imzaları Task 3 ↔ Task 5/6 kullanımı tutarlı ✓.
