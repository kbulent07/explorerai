# Config-Driven Plugin Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FaceZoom kamera işlem hattının analiz/çıktı aşamalarını (tanıma, sayım, zoom, overlay) YAML ile tanımlanan bir modül zincirine çevirmek; algılama çekirdeği sabit kalır.

**Architecture:** `CameraWorker.process()` iki faza ayrılır: (1) SABİT algılama çekirdeği bir `ctx` sözlüğü doldurur (kareler, ölçek, faces, tracks, finished best-shot'lar); (2) config'teki modül zinciri önce tüm `process()` (analiz), sonra tüm `draw()` (display) geçişini yapar. Modüller dotted-path ile importlib'den yüklenir; `pipeline:` tanımsızsa bugünkü davranışı üreten varsayılan zincir sentezlenir. Davranış-korumalı refactor.

**Tech Stack:** Python 3.10+, OpenCV (cv2), mevcut FaceZoom modülleri (framing.FrameTransformer, tracking, recognition.RecognitionPipeline, counting.LineCrossingCounter/CountingStore, worker._NameResolver), pytest.

## Global Constraints

- Kod yorumları ASCII-translit Türkçe (diakritiksiz); belgeler tam Türkçe. (Repo konvansiyonu)
- Davranış-koruma: `pipeline:` tanımsız mevcut `config.yaml` ile çıktı birebir aynı kalmalı.
- Modül hatası kamerayı/zinciri düşürmemeli; yakalanıp loglanmalı.
- Mevcut singleton'lar (`RecognitionPipeline`, `RecentFaceStore`, `CountingStore`, `LineCrossingCounter`, `_NameResolver`, `FrameTransformer`) aynen kullanılır; yalnız çağrı yerleri modüllere taşınır.
- Her yeni Python dosyası tek sorumluluk taşır; `modules/` yeni pakettir.
- Testler `venv` ile çalıştırılır: `venv/Scripts/python.exe -m pytest` (Windows) / `venv/bin/python -m pytest`.

---

### Task 1: PipelineModule tabanı + modül yükleyici

**Files:**
- Create: `pipeline.py`
- Test: `tests/test_pipeline_loader.py`

**Interfaces:**
- Produces:
  - `class PipelineModule` — metodlar: `setup(self, config, camera, services)`, `process(self, ctx)`, `draw(self, ctx)`, `finalize(self)` (hepsi no-op varsayılan).
  - `load_module(path: str, config: dict, camera: str, services: dict) -> PipelineModule | None` — `path` "modul.altyol:SinifAdi" biçimi; yükleyip `setup` çağırır; hata olursa `None` döner + loglar.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_loader.py
import pipeline


class _Dummy(pipeline.PipelineModule):
    def __init__(self):
        self.setup_args = None
    def setup(self, config, camera, services):
        self.setup_args = (config, camera, services)


def test_base_metodlari_noop():
    m = pipeline.PipelineModule()
    # taban metodlar cagrilabilir ve hicbir sey dondurmez (no-op)
    assert m.process({}) is None
    assert m.draw({}) is None
    assert m.setup({}, "Kam", {}) is None
    assert m.finalize() is None


def test_load_module_yukler_ve_setup_cagirir():
    m = pipeline.load_module("tests.helpers_pipe:GoodModule",
                             {"k": 1}, "Kam", {"svc": 2})
    assert m is not None
    assert m.setup_args == ({"k": 1}, "Kam", {"svc": 2})


def test_load_module_bozuk_yol_none_doner():
    assert pipeline.load_module("yok.modul:Sinif", {}, "Kam", {}) is None
    assert pipeline.load_module("bozukbiciim", {}, "Kam", {}) is None
```

Create test helper:

```python
# tests/helpers_pipe.py
import pipeline


class GoodModule(pipeline.PipelineModule):
    def __init__(self):
        self.setup_args = None
    def setup(self, config, camera, services):
        self.setup_args = (config, camera, services)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_pipeline_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline.py
# -----------------------------------------------------------------------------
# Config-driven islem hatti: kamera analiz/cikti asamalari (tanima, sayim, zoom,
# overlay) YAML'daki modul zinciri olarak calisir. Algilama cekirdegi (worker)
# ctx sozlugunu doldurur; buradaki moduller onu isler/cizer.
# -----------------------------------------------------------------------------

import importlib
import logging

log = logging.getLogger("facezoom.pipeline")


class PipelineModule:
    """Islem hatti modulu tabani. Alt siniflar ihtiyac duydugu metodu uygular;
    hepsi opsiyonel (taban no-op saglar)."""

    def setup(self, config, camera, services):
        """Bir kez kurulum. config: tam config dict; camera: kamera adi;
        services: uygulama singleton'lari (on_capture, line_counter, ...)."""
        return None

    def process(self, ctx):
        """ANALIZ fazi: ctx'i gunceller (embedding, event...). Ciktiyi CIZMEZ."""
        return None

    def draw(self, ctx):
        """DISPLAY fazi: ctx['output'] uzerinde calisir (annotate/transform)."""
        return None

    def finalize(self):
        """Kapanista kaynak birak (thread, dedektor...)."""
        return None


def load_module(path, config, camera, services):
    """'paket.modul:SinifAdi' -> ornek. Yukleyip setup() cagirir. Hata olursa
    None doner (zincir kalanla devam etsin diye). Birkac constructor imzasi denenir."""
    if not isinstance(path, str) or ":" not in path:
        log.warning("Gecersiz modul yolu (paket.modul:Sinif bekleniyor): %r", path)
        return None
    mod_path, _, cls_name = path.partition(":")
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
    except Exception:
        log.warning("Modul yuklenemedi: %s (atlaniyor)", path, exc_info=True)
        return None
    # Birkac constructor imzasi dene: (config, camera), (config), ()
    inst = None
    for args in ((config, camera), (config,), ()):
        try:
            inst = cls(*args)
            break
        except TypeError:
            continue
        except Exception:
            log.warning("Modul kurucu hatasi: %s (atlaniyor)", path, exc_info=True)
            return None
    if inst is None:
        log.warning("Modul kurucusu hicbir imzayla kurulamadi: %s", path)
        return None
    try:
        inst.setup(config, camera, services)
    except Exception:
        log.warning("Modul setup hatasi: %s (atlaniyor)", path, exc_info=True)
        return None
    return inst
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_pipeline_loader.py -v`
Expected: PASS (4 test)

- [ ] **Step 5: Commit**

```bash
git add pipeline.py tests/test_pipeline_loader.py tests/helpers_pipe.py
git commit -m "feat(pipeline): PipelineModule tabani + dotted-path yukleyici"
```

---

### Task 2: build_pipeline (varsayılan sentez) + Pipeline.run (iki geçiş + hata izolasyonu)

**Files:**
- Modify: `pipeline.py`
- Test: `tests/test_pipeline_build_run.py`

**Interfaces:**
- Consumes: `PipelineModule`, `load_module` (Task 1).
- Produces:
  - `class Pipeline` — `__init__(self, modules: list[PipelineModule])`; `run(self, ctx)` (önce tüm `process()`, sonra tüm `draw()`; her modül hatası yakalanır, throttled loglanır, o kare için atlanır); `finalize(self)` (her modülün `finalize()`'i).
  - `build_pipeline(config, camera, services, *, is_counting=False) -> Pipeline` — `config["pipeline"]` (kamera override öncelikli) varsa o listeden; yoksa varsayılan zinciri sentezler.
  - Varsayılan sentez kuralı: `["modules.recognition:RecognitionModule"]` (her zaman) + (`is_counting` ise `"modules.counting:CountingModule"`) + (`config.get("zoom_enabled", True)` ise `"modules.zoom:ZoomModule"`) + (`config.get("debug_overlay", True)` ise `"modules.overlay:OverlayModule"`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_build_run.py
import pipeline


class _Rec(pipeline.PipelineModule):
    def __init__(self):
        self.calls = []
    def process(self, ctx):
        self.calls.append("p")
        ctx.setdefault("order", []).append("rec.process")
    def draw(self, ctx):
        ctx.setdefault("order", []).append("rec.draw")


class _Boom(pipeline.PipelineModule):
    def process(self, ctx):
        raise RuntimeError("patladim")
    def draw(self, ctx):
        raise RuntimeError("cizerken patladim")


def test_run_iki_gecis_sirasi():
    a, b = _Rec(), _Rec()
    p = pipeline.Pipeline([a, b])
    ctx = {}
    p.run(ctx)
    # once tum process (liste sirasi), sonra tum draw (liste sirasi)
    assert ctx["order"] == ["rec.process", "rec.process", "rec.draw", "rec.draw"]


def test_run_modul_hatasi_zinciri_kesmez():
    good = _Rec()
    p = pipeline.Pipeline([_Boom(), good])
    ctx = {}
    p.run(ctx)   # Boom patlar ama yakalanir
    assert "rec.process" in ctx["order"]   # good yine calisti


def test_build_varsayilan_zincir_sentezi():
    cfg = {"zoom_enabled": True, "debug_overlay": False}
    p = pipeline.build_pipeline(cfg, "Kam", {}, is_counting=True)
    names = [type(m).__name__ for m in p.modules]
    assert names == ["RecognitionModule", "CountingModule", "ZoomModule"]


def test_build_acik_liste_kullanir():
    cfg = {"pipeline": ["tests.helpers_pipe:GoodModule"]}
    p = pipeline.build_pipeline(cfg, "Kam", {})
    assert [type(m).__name__ for m in p.modules] == ["GoodModule"]


def test_build_kamera_override_onceligi():
    cfg = {"pipeline": ["modules.overlay:OverlayModule"],
           "_camera_pipeline": ["tests.helpers_pipe:GoodModule"]}
    # kamera-bazli liste services uzerinden verilir (asagida imzaya bkz.)
    p = pipeline.build_pipeline(cfg, "Kam", {}, camera_pipeline=cfg["_camera_pipeline"])
    assert [type(m).__name__ for m in p.modules] == ["GoodModule"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_pipeline_build_run.py -v`
Expected: FAIL with `AttributeError: module 'pipeline' has no attribute 'Pipeline'`

- [ ] **Step 3: Write minimal implementation**

Append to `pipeline.py`:

```python
class Pipeline:
    """Modul zinciri. run(ctx): once tum process (analiz), sonra tum draw
    (display), liste sirasiyla. Bir modul patlarsa yakalanir + throttled loglanir;
    o kare icin o modul atlanir, zincir/canli akis kesilmez."""

    def __init__(self, modules):
        self.modules = list(modules)
        self._err_counts = {}   # id(modul) -> hata sayaci (log throttle)

    def _run_one(self, phase, fn, ctx):
        try:
            fn(ctx)
        except Exception:
            key = (id(fn.__self__), phase)
            n = self._err_counts.get(key, 0) + 1
            self._err_counts[key] = n
            if n == 1 or n % 50 == 0:
                log.warning("Modul %s.%s hatasi (#%d, kare atlaniyor)",
                            type(fn.__self__).__name__, phase, n, exc_info=True)

    def run(self, ctx):
        for m in self.modules:
            self._run_one("process", m.process, ctx)
        for m in self.modules:
            self._run_one("draw", m.draw, ctx)

    def finalize(self):
        for m in self.modules:
            try:
                m.finalize()
            except Exception:
                log.warning("Modul %s finalize hatasi", type(m).__name__,
                            exc_info=True)


# Varsayilan zincir: pipeline: tanimsizsa bugunku davranisi uretir.
def _default_chain(config, is_counting):
    chain = ["modules.recognition:RecognitionModule"]
    if is_counting:
        chain.append("modules.counting:CountingModule")
    if config.get("zoom_enabled", True):
        chain.append("modules.zoom:ZoomModule")
    if config.get("debug_overlay", True):
        chain.append("modules.overlay:OverlayModule")
    return chain


def build_pipeline(config, camera, services, *, is_counting=False,
                   camera_pipeline=None):
    """Modul zincirini kur. Oncelik: camera_pipeline (kamera-bazli) > config['pipeline']
    (global) > varsayilan sentez. services: modullere setup ile verilecek singleton'lar."""
    if camera_pipeline is not None:
        paths = camera_pipeline
    elif isinstance(config.get("pipeline"), (list, tuple)):
        paths = list(config["pipeline"])
    else:
        paths = _default_chain(config, is_counting)
    modules = []
    for p in paths:
        m = load_module(p, config, camera, services)
        if m is not None:
            modules.append(m)
    return Pipeline(modules)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_pipeline_build_run.py -v`
Expected: PASS (5 test). (Not: `test_build_varsayilan_zincir_sentezi` gerçek modül sınıflarını yükler; bu test Task 3–6 tamamlanınca yeşile döner. Bu aşamada `modules.*` yüklenemediği için `p.modules` boş kalır → testi Task 6 sonrası doğrula. Şimdilik `test_build_acik_liste_kullanir`, `test_run_*`, `test_build_kamera_override_onceligi` geçmeli.)

> **Not (bağımlılık sırası):** `test_build_varsayilan_zincir_sentezi` gerçek `modules.recognition` vb. gerektirir. O testi bu adımda `@pytest.mark.skip(reason="modules Task 3-6'da eklenecek")` ile işaretle; Task 6 sonrası skip'i kaldır.

- [ ] **Step 5: Commit**

```bash
git add pipeline.py tests/test_pipeline_build_run.py
git commit -m "feat(pipeline): Pipeline.run (iki gecis + hata izolasyonu) + build_pipeline"
```

---

### Task 3: OverlayModule (draw)

**Files:**
- Create: `modules/__init__.py` (boş), `modules/overlay.py`
- Test: `tests/test_module_overlay.py`

**Interfaces:**
- Consumes: `PipelineModule`; `ctx` alanları: `output` (ndarray), `camera` (str), `fps` (float), `face_present` (bool), `zoomed` (bool), `debug_overlay` (bool, config'ten setup'ta alınır).
- Produces: `class OverlayModule(PipelineModule)` — `draw(ctx)` çizim yapar.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_module_overlay.py
import numpy as np
from modules.overlay import OverlayModule


def test_overlay_output_uzerine_cizer():
    m = OverlayModule()
    m.setup({"debug_overlay": True}, "Kam", {})
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    ctx = {"output": frame, "camera": "Kam", "fps": 12.3,
           "face_present": True, "zoomed": False}
    before = frame.copy()
    m.draw(ctx)
    # ust seride (overlay bandi) pikseller degismis olmali
    assert not np.array_equal(before[:28], ctx["output"][:28])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_module_overlay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules'`

- [ ] **Step 3: Write minimal implementation**

```python
# modules/__init__.py
# (bos: FaceZoom islem hatti modulleri paketi)
```

```python
# modules/overlay.py
# Debug overlay modulu: FPS + yuz durumu + kamera adi + ZOOM etiketi. draw() ile
# ctx['output'] uzerine cizer. (worker._draw_overlay'in modul karsiligi.)

import cv2 as cv

from pipeline import PipelineModule


class OverlayModule(PipelineModule):
    def setup(self, config, camera, services):
        self.enabled = bool(config.get("debug_overlay", True))

    def draw(self, ctx):
        if not getattr(self, "enabled", True):
            return
        frame = ctx.get("output")
        if frame is None:
            return
        face_present = ctx.get("face_present", False)
        status = "YUZ VAR" if face_present else "yuz yok"
        color = (0, 220, 0) if face_present else (0, 165, 255)
        conn = "" if ctx.get("connected", True) else "  [BAGLANTI YOK]"
        text = f"{ctx.get('camera','')}  FPS:{ctx.get('fps',0.0):4.1f}  {status}{conn}"
        cv.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
        cv.putText(frame, text, (8, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if ctx.get("zoomed"):
            cv.putText(frame, "ZOOM", (frame.shape[1] - 90, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_module_overlay.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modules/__init__.py modules/overlay.py tests/test_module_overlay.py
git commit -m "feat(modules): OverlayModule (debug overlay draw)"
```

---

### Task 4: ZoomModule (draw + set_enabled)

**Files:**
- Create: `modules/zoom.py`
- Test: `tests/test_module_zoom.py`

**Interfaces:**
- Consumes: `PipelineModule`; `framing.FrameTransformer`; `worker.scale_bbox`; `ctx` alanları: `output`, `hires_dims=(hw,hh)`, `faces` (bbox detect uzayı), `scale=(sx,sy)`, `now`.
- Produces: `class ZoomModule(PipelineModule)` — `draw(ctx)` (transform + `ctx["zoomed"]` yazar), `set_enabled(bool)` (canlı ac/kapat).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_module_zoom.py
import numpy as np
from modules.zoom import ZoomModule


def test_zoom_kapaliyken_output_degismez():
    m = ZoomModule()
    m.setup({"zoom_enabled": False, "output_size": [200, 100]}, "Kam", {})
    frame = np.full((100, 200, 3), 7, dtype=np.uint8)
    ctx = {"output": frame, "hires_dims": (200, 100), "faces": [], "scale": (1.0, 1.0), "now": 1.0}
    m.draw(ctx)
    assert ctx.get("zoomed") in (False, None)
    assert np.array_equal(ctx["output"], frame)   # dokunmadi


def test_zoom_acikken_zoomed_bayragini_yazar():
    m = ZoomModule()
    m.setup({"zoom_enabled": True, "zoom_factor": 2.0}, "Kam", {})
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    ctx = {"output": frame, "hires_dims": (200, 100),
           "faces": [{"bbox": (80, 40, 40, 20)}], "scale": (1.0, 1.0), "now": 1.0}
    m.draw(ctx)
    assert "zoomed" in ctx            # bayrak set edildi (True/False FrameTransformer'a bagli)
    assert ctx["output"].shape[0] > 0


def test_set_enabled_calisir():
    m = ZoomModule()
    m.setup({"zoom_enabled": True}, "Kam", {})
    m.set_enabled(False)
    assert m.enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_module_zoom.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.zoom'`

- [ ] **Step 3: Write minimal implementation**

```python
# modules/zoom.py
# Canli dijital pan-zoom modulu: ctx['output']'u en buyuk yuze odakli olcekler.
# FrameTransformer durumu (EMA/hold) modul icinde tutulur. (worker'daki zoom
# blogunun modul karsiligi.) set_enabled ile canli ac/kapat (LiveManager.set_zoom).

from pipeline import PipelineModule
from framing import FrameTransformer
from worker import scale_bbox


class ZoomModule(PipelineModule):
    def setup(self, config, camera, services):
        self.enabled = bool(config.get("zoom_enabled", True))
        self._zoom_factor = config.get("zoom_factor", 2.5)
        self._smoothing = config.get("smoothing", 0.15)
        self._hold = config.get("hold_seconds", 1.5)
        self._tf = None

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)

    def _ensure_tf(self, fw, fh):
        if self._tf is None:
            self._tf = FrameTransformer(fw, fh, zoom_factor=self._zoom_factor,
                                        smoothing=self._smoothing,
                                        hold_seconds=self._hold)
        elif (self._tf.frame_width, self._tf.frame_height) != (fw, fh):
            self._tf.update_size(fw, fh)

    def draw(self, ctx):
        if not self.enabled:
            ctx["zoomed"] = False
            return
        frame = ctx.get("output")
        if frame is None:
            return
        hw, hh = ctx.get("hires_dims", (frame.shape[1], frame.shape[0]))
        self._ensure_tf(hw, hh)
        target = None
        faces = ctx.get("faces") or []
        if faces:
            sx, sy = ctx.get("scale", (1.0, 1.0))
            largest = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
            target = scale_bbox(largest["bbox"], sx, sy)
        out, zoomed = self._tf.transform(frame, target, now=ctx.get("now"))
        ctx["output"] = out
        ctx["zoomed"] = zoomed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_module_zoom.py -v`
Expected: PASS (3 test)

- [ ] **Step 5: Commit**

```bash
git add modules/zoom.py tests/test_module_zoom.py
git commit -m "feat(modules): ZoomModule (pan-zoom draw + set_enabled)"
```

---

### Task 5: CountingModule (process)

**Files:**
- Create: `modules/counting.py`
- Test: `tests/test_module_counting.py`

**Interfaces:**
- Consumes: `PipelineModule`; `worker.scale_bbox`, `worker.crop_with_margin`; `services` anahtarları: `line_counter` (LineCrossingCounter), `counting_store` (CountingStore), `name_resolver` (_NameResolver | None), `manager` (best_crop için track erişimi — `manager.tracks.get(tid).best_crop`); `ctx` alanları: `tracks`, `detect_dims`, `tid_face`, `scale`, `hires_frame`, `now`, `camera`.
- Produces: `class CountingModule(PipelineModule)` — `process(ctx)`: çizgi-geçiş → event → `counting_store.record` (isimsiz) → varsa `name_resolver.submit`. (worker._run_counting'in birebir taşınması.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_module_counting.py
import numpy as np
from modules.counting import CountingModule
from counting import LineCrossingCounter, CountingStore


class _FakeMgr:
    tracks = {}


def _ctx(tracks):
    return {"tracks": tracks, "detect_dims": (200, 200), "tid_face": {},
            "scale": (1.0, 1.0), "hires_frame": np.zeros((200, 200, 3), np.uint8),
            "now": 1.0, "camera": "Kam"}


def test_counting_gecis_kaydeder():
    lc = LineCrossingCounter(line=(0.0, 0.5, 1.0, 0.5))
    store = CountingStore()
    m = CountingModule()
    m.setup({}, "Kam", {"line_counter": lc, "counting_store": store,
                        "name_resolver": None, "manager": _FakeMgr()})
    m.process(_ctx([(1, (90, 40, 20, 20))]))    # ust taraf
    m.process(_ctx([(1, (90, 140, 20, 20))]))   # cizgiyi gecti
    c = store.counts()
    assert c["in"] + c["out"] == 1


def test_counting_sayac_yoksa_noop():
    m = CountingModule()
    m.setup({}, "Kam", {"line_counter": None, "counting_store": None,
                        "name_resolver": None, "manager": _FakeMgr()})
    m.process(_ctx([(1, (90, 140, 20, 20))]))   # patlamamali
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_module_counting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.counting'`

- [ ] **Step 3: Write minimal implementation**

```python
# modules/counting.py
# Giris/cikis sayim modulu: ctx['tracks']'te cizgi-gecis tespiti -> CountingStore'a
# isimsiz olay + (varsa) asenkron isim cozumu. worker._run_counting'in modul hali.

import logging

import cv2 as cv

from pipeline import PipelineModule
from worker import scale_bbox, crop_with_margin

log = logging.getLogger("facezoom.modules.counting")


class CountingModule(PipelineModule):
    def setup(self, config, camera, services):
        self.line_counter = services.get("line_counter")
        self.counting_store = services.get("counting_store")
        self.name_resolver = services.get("name_resolver")
        self.manager = services.get("manager")

    def process(self, ctx):
        if self.line_counter is None or self.counting_store is None:
            return
        tracks = ctx.get("tracks") or []
        dims = ctx.get("detect_dims")
        self.counting_store.note(len(tracks))
        events = self.line_counter.update(tracks, dims)
        if not events:
            return
        bbox_by_tid = dict(tracks)
        tid_face = ctx.get("tid_face") or {}
        sx, sy = ctx.get("scale", (1.0, 1.0))
        hires = ctx.get("hires_frame")
        now = ctx.get("now")
        for tid, direction in events:
            jpeg = None
            tr = self.manager.tracks.get(tid) if self.manager is not None else None
            fcrop = getattr(tr, "best_crop", None) if tr is not None else None
            if fcrop is None:
                face = tid_face.get(tid)
                if face is not None:
                    fcrop = crop_with_margin(hires, scale_bbox(face["bbox"], sx, sy), 0.3)
            crop = fcrop
            if crop is None:
                dbbox = bbox_by_tid.get(tid)
                if dbbox is not None:
                    crop = crop_with_margin(hires, scale_bbox(dbbox, sx, sy), 0.1)
            if crop is not None:
                ok, buf = cv.imencode(".jpg", crop, [cv.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    jpeg = buf.tobytes()
            eid = self.counting_store.record(direction, ts=now, name=None,
                                             camera=ctx.get("camera"), jpeg=jpeg)
            if eid is not None and fcrop is not None and self.name_resolver is not None:
                self.name_resolver.submit(eid, fcrop, ctx.get("camera"), now)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_module_counting.py -v`
Expected: PASS (2 test)

- [ ] **Step 5: Commit**

```bash
git add modules/counting.py tests/test_module_counting.py
git commit -m "feat(modules): CountingModule (cizgi-gecis sayimi process)"
```

---

### Task 6: RecognitionModule (process — web yakalama-sink'i)

**Files:**
- Create: `modules/recognition.py`
- Test: `tests/test_module_recognition.py`

**Interfaces:**
- Consumes: `PipelineModule`; `services["on_capture"]` (callable | None) — imza: `(camera, crop, bbox, quality, first_seen, last_seen, best_time)`; `ctx` alanları: `finished` (list[Track]), `camera`.
- Produces: `class RecognitionModule(PipelineModule)` — `process(ctx)`: `ctx["finished"]` her track için `on_capture` çağırır. `on_capture` None ise no-op. (webui `_on_capture`'ın tetiklenme yolu; `_on_capture` içeriği webui'de aynen kalır.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_module_recognition.py
from modules.recognition import RecognitionModule


class _Track:
    def __init__(self, tid):
        self.best_crop = f"crop{tid}"
        self.best_bbox = (0, 0, 10, 10)
        self.best_score = 0.9
        self.first_seen = 1.0
        self.last_seen = 2.0
        self.best_time = 1.5


def test_recognition_finished_icin_on_capture_cagirir():
    calls = []
    m = RecognitionModule()
    m.setup({}, "Kam", {"on_capture": lambda *a: calls.append(a)})
    m.process({"finished": [_Track(1), _Track(2)], "camera": "Kam"})
    assert len(calls) == 2
    assert calls[0][0] == "Kam" and calls[0][1] == "crop1"


def test_recognition_on_capture_yoksa_noop():
    m = RecognitionModule()
    m.setup({}, "Kam", {"on_capture": None})
    m.process({"finished": [_Track(1)], "camera": "Kam"})   # patlamamali
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_module_recognition.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.recognition'`

- [ ] **Step 3: Write minimal implementation**

```python
# modules/recognition.py
# Web yakalama-sink modulu: biten gorunumlerin (ctx['finished']) best-shot'larini
# uygulama on_capture geri-cagrimina iletir. on_capture (webui) tanima acikken
# RecognitionPipeline'a submit eder, degilse dogrudan RECENT'e yazar -> mevcut
# _on_capture dallanmasi webui'de aynen korunur. CLI'da on_capture None -> no-op.

from pipeline import PipelineModule


class RecognitionModule(PipelineModule):
    def setup(self, config, camera, services):
        self.on_capture = services.get("on_capture")

    def process(self, ctx):
        if self.on_capture is None:
            return
        for tr in ctx.get("finished") or []:
            self.on_capture(ctx.get("camera"), tr.best_crop, tr.best_bbox,
                            tr.best_score, tr.first_seen, tr.last_seen, tr.best_time)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_module_recognition.py -v`
Expected: PASS (2 test)

- [ ] **Step 5: Remove the skip on the default-chain test (Task 2) and verify**

`tests/test_pipeline_build_run.py` içindeki `test_build_varsayilan_zincir_sentezi` üzerindeki `@pytest.mark.skip` işaretini kaldır.

Run: `venv/Scripts/python.exe -m pytest tests/test_pipeline_build_run.py::test_build_varsayilan_zincir_sentezi -v`
Expected: PASS (modules.recognition/counting/zoom artık yüklenebiliyor)

- [ ] **Step 6: Commit**

```bash
git add modules/recognition.py tests/test_module_recognition.py tests/test_pipeline_build_run.py
git commit -m "feat(modules): RecognitionModule (web yakalama-sink) + varsayilan zincir testi acildi"
```

---

### Task 7: worker.py entegrasyonu — process() iki faz + pipeline

**Files:**
- Modify: `worker.py` (`CameraWorker.__init__`, `process()`, `finalize()`, `set_zoom()`)
- Test: `tests/test_worker_pipeline_integration.py`

**Interfaces:**
- Consumes: `pipeline.build_pipeline`, tüm `modules.*`.
- Produces: `CameraWorker` artık `process()` içinde `ctx` doldurup `self._pipeline.run(ctx)` çağırır; `self._pipeline` `__init__`'te `build_pipeline` ile kurulur; `set_zoom` ZoomModule'e delege eder; `finalize` `self._pipeline.finalize()` çağırır.

**Detay — `services` sözlüğü (`__init__`'te):**
```python
services = {
    "on_capture": self.on_capture,
    "line_counter": self.line_counter,
    "counting_store": self.counting_store,
    "name_resolver": self._name_resolver,   # mevcut _NameResolver (Task: bu oturumda eklendi)
    "manager": self.manager,
}
```

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_worker_pipeline_integration.py
# Varsayilan zincirle CameraWorker eski davranisi verir: biten track on_capture'a
# gider; zoom kapaliyken output degismeden gecer. (Kamera/mediapipe olmadan,
# process()'in pipeline entegrasyonunu izole dogrular.)
import numpy as np
import worker as W


def test_ctx_fill_ve_pipeline_run(monkeypatch):
    # Sahte kamera: hep ayni hires kareyi verir, detect yok
    class _Cam:
        name = "Kam"
        connected = True
        def read_detect(self): return None, 0
        def read_hires(self): return np.zeros((120, 160, 3), np.uint8), 1
    captured = []
    cfg = {"detector_backend": "mediapipe", "zoom_enabled": False,
           "debug_overlay": False, "recognition_enabled": True,
           "output_size": [160, 120], "detect_interval": 1}
    w = W.CameraWorker(_Cam(), cfg, on_capture=lambda *a: captured.append(a))
    # pipeline kuruldu mu + RecognitionModule iceriyor mu
    names = [type(m).__name__ for m in w._pipeline.modules]
    assert "RecognitionModule" in names
    out = w.process()
    assert out is not None and out.shape[:2] == (120, 160)   # output_size'a resize
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_worker_pipeline_integration.py -v`
Expected: FAIL (`w._pipeline` yok / process eski yapıda)

- [ ] **Step 3: Refactor `CameraWorker.__init__` — pipeline kur**

`__init__` sonuna (mevcut `_name_resolver` ve `_diag` kurulumundan sonra) ekle:

```python
        # --- config-driven islem hatti (analiz/cikti asamalari) ---
        import pipeline as _pipeline_mod
        services = {
            "on_capture": self.on_capture,
            "line_counter": self.line_counter,
            "counting_store": self.counting_store,
            "name_resolver": self._name_resolver,
            "manager": self.manager,
        }
        cam_pipe = None
        # kamera-bazli override: config'te bu kameranin blogunda pipeline: varsa
        cam_pipe = config.get("pipeline") if isinstance(config.get("pipeline"), (list, tuple)) else None
        self._pipeline = _pipeline_mod.build_pipeline(
            config, camera.name, services,
            is_counting=(self.line_counter is not None),
            camera_pipeline=cam_pipe if cam_pipe is not None else None,
        )
```

> **Not:** Kamera-bazlı override tam desteği (kamera bloğundaki `pipeline:`) Aşama 3'te config merge ile netleşecek; Aşama 1'de global `config["pipeline"]` yeterli. Yukarıdaki `cam_pipe` global ile aynı olduğundan `build_pipeline` global listeyi kullanır.

- [ ] **Step 4: Refactor `process()` — çekirdek ctx doldurur, pipeline çalışır**

`process()` içinde, mevcut `_record_best`/`_run_counting`/zoom/overlay/resize bloklarını KALDIR ve yerine `ctx` doldurup pipeline çalıştır. `run_detect` bloğu (tespit+track+best-shot kaydı) AYNEN kalır; yalnız `_run_counting` çağrısı kalkar (CountingModule'e taşındı). `collect_finished` sonrası:

```python
        # --- bitmiss gorunumler: DB (CLI) core'da; RECENT (web) modulde ---
        finished = self.manager.collect_finished(now)
        if self.db is not None:
            for tr in finished:
                self.db.save_capture(
                    camera_name=self.camera.name, crop_bgr=tr.best_crop,
                    quality_score=tr.best_score, first_seen=tr.first_seen,
                    last_seen=tr.last_seen, best_time=tr.best_time)
        self._diag["emitted"] += len(finished)

        # FPS
        dt = now - self._fps_t
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._fps_t = now

        # --- ctx doldur + islem hatti (analiz + display) ---
        ctx = {
            "camera": self.camera.name, "now": now,
            "detect_frame": det_source, "detect_dims": (dw, dh),
            "hires_frame": hires_frame, "hires_dims": (hw, hh),
            "scale": (sx, sy), "faces": self._faces,
            "tracks": tracks, "tid_face": tid_face, "finished": finished,
            "run_detect": run_detect, "output": hires_frame,
            "fps": self._fps, "face_present": len(self._faces) > 0,
            "connected": self.camera.connected, "zoomed": False,
        }
        self._pipeline.run(ctx)
        output = ctx["output"]

        # cikti boyutuna olcekle (bicimsiz output_size -> guvenli varsayilan)
        osz = self.config.get("output_size", [1280, 720])
        if not (isinstance(osz, (list, tuple)) and len(osz) == 2):
            osz = [1280, 720]
        out_w, out_h = int(osz[0]), int(osz[1])
        if (output.shape[1], output.shape[0]) != (out_w, out_h):
            output = cv.resize(output, (out_w, out_h), interpolation=cv.INTER_LINEAR)

        # periyodik teshis (mevcut _diag blogu aynen korunur)
        # ... (mevcut kod)
        return output
```

> `tracks` ve `tid_face` değişkenleri `run_detect` bloğunda tanımlı; `run_detect` False olduğu karelerde önceki değerleri kullanmak için `__init__`'te `self._tracks=[]`, `self._tid_face={}` tut ve `run_detect` bloğunda bunları güncelle, ctx'e `self._tracks`/`self._tid_face` koy. (Aksi halde detect atlanan karede NameError olur.)

Bu nedenle önce `__init__`'e ekle: `self._tracks = []` ve `self._tid_face = {}`. `run_detect` bloğunda `tracks`/`tid_face` yerine `self._tracks`/`self._tid_face` doldur; ctx'te onları kullan. `det_source`,`dw`,`dh`,`sx`,`sy`,`hw`,`hh` zaten her karede hesaplanıyor.

- [ ] **Step 5: `set_zoom` ve `finalize`'i güncelle**

```python
    def set_zoom(self, enabled):
        """Canli pan-zoom'u ac/kapat: ZoomModule varsa ona delege et."""
        self.zoom_enabled = bool(enabled)
        for m in self._pipeline.modules:
            if hasattr(m, "set_enabled") and type(m).__name__ == "ZoomModule":
                m.set_enabled(enabled)

    def finalize(self):
        now = time.time()
        for tr in self.manager.flush_all(now):
            self._emit_capture(tr)
        if self._name_resolver is not None:
            self._name_resolver.stop()
        self._pipeline.finalize()
        self.tracker.close()
```

> `_emit_capture` artık yalnız `flush_all` (kapanış) için gerekli; web'de RECENT'e yazımı korumak adına `_emit_capture`'ın `on_capture` dalı KORUNUR (kapanışta biten görünümler de RECENT'e düşsün). Yani `_emit_capture` içeriği DEĞİŞMEZ; yalnız `process()` içindeki her-kare `_emit_capture` çağrısı kalkar (yerini core-db + RecognitionModule alır).

- [ ] **Step 6: Run integration + full suite**

Run: `venv/Scripts/python.exe -m pytest tests/test_worker_pipeline_integration.py -v`
Expected: PASS

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: tüm testler PASS (mevcut + yeni). `test_worker_backend_selection` yeşil kalmalı.

- [ ] **Step 7: Commit**

```bash
git add worker.py tests/test_worker_pipeline_integration.py
git commit -m "feat(worker): process() iki faz -> config-driven islem hatti (davranis-korumali)"
```

---

### Task 8: config.example.yaml + DOCKER.md belgeleme

**Files:**
- Modify: `config.example.yaml`, `DOCKER.md`

**Interfaces:** yok (dokümantasyon).

- [ ] **Step 1: config.example.yaml'a pipeline bölümü ekle**

`# ---- algilama / takip ----` bölümünün üstüne ekle:

```yaml
# ---- islem hatti (pipeline) ----
# Opsiyonel. TANIMSIZSA bugunku davranis uretilir (recognition + [sayim kamerasinda
# counting] + [zoom_enabled ise zoom] + [debug_overlay ise overlay]).
# Tanimlarsan analiz/cikti asamalarinin SIRASINI ve KUMESINI sen belirlersin;
# "yeni ozellik = yeni modul dosyasi + buraya bir satir". Cizim sirasi = liste sirasi.
# pipeline:
#   - modules.recognition:RecognitionModule
#   - modules.counting:CountingModule
#   - modules.zoom:ZoomModule
#   - modules.overlay:OverlayModule
```

- [ ] **Step 2: DOCKER.md'ye kısa not ekle**

"Notlar" bölümüne ekle:

```markdown
- **İşlem hattı (pipeline):** Kamera analiz/çıktı aşamaları (tanıma, sayım, zoom,
  overlay) `config.yaml > pipeline` ile modül listesi olarak yeniden sıralanabilir.
  Tanımsızsa varsayılan (mevcut) davranış üretilir. Yeni modül = `modules/` altında
  yeni dosya + listeye bir satır.
```

- [ ] **Step 3: Commit**

```bash
git add config.example.yaml DOCKER.md
git commit -m "docs(pipeline): config-driven pipeline kullanimi belgelendi"
```

---

## Self-Review Notu (yazan için)

- **Spec kapsamı:** Task 1–2 = framework (§2,§3,§8); Task 3–6 = modül seti (§5); Task 7 = entegrasyon + geriye-uyum (§6,§7,§2); Task 8 = config formatı (§6). Test planı (§10) her task'ın adımlarına dağıtıldı. Kabul kriterleri (§12) Task 7 tam suite + Task 6 varsayılan-zincir testiyle karşılanır.
- **Bağımlılık sırası:** `modules/*` `pipeline`'a ve `worker`'ın `scale_bbox`/`crop_with_margin`/`FrameTransformer`'ına bağlı; import döngüsü riski: `modules.zoom`/`modules.counting` `worker`'dan import eder, `worker` `pipeline`'dan (modülleri değil) import eder → `worker` modülleri yalnız `build_pipeline` çağrısıyla (çalışma anında importlib) yükler, tepe-seviye `import modules...` yapmaz → döngü yok. Bunu Task 7'de doğrula (import hatası çıkarsa `scale_bbox`/`crop_with_margin`'i `pipeline`'a değil ayrı bir `geometry.py`'ye taşımayı değerlendir — ama önce döngüsüz varsayımı test et).
- **Tip tutarlılığı:** `on_capture` imzası, `Track` alanları (`best_crop/best_bbox/best_score/first_seen/last_seen/best_time`), `LineCrossingCounter.update(tracks,dims)->events`, `CountingStore.record(...)->eid`, `_NameResolver.submit(eid,crop,camera,ts)`, `FrameTransformer(fw,fh,...).transform(frame,target,now)->(out,zoomed)` — hepsi mevcut koddan doğrulandı ve task'lar arası tutarlı kullanıldı.
