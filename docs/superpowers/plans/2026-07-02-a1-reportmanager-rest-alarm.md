# Faz A1 — ReportManager (REST Alarm/Rapor) Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** AiEye olaylarını (sayım geçişleri, biten görünümler) blackboard `ctx["events"]` üzerinden toplayıp bir daemon kuyruğuyla uzak REST API'ye güvenilir gönderen raporlama katmanı.

**Architecture:** İki katman: (1) `reporting.py` — pipeline'dan habersiz `ReportManager` (deque kuyruk + daemon gönderici + rate-limit + offline disk snapshot); (2) `modules/reporting.py:ReportingModule` — `ctx["events"]`'i filtreleyip `ReportManager.send()`'e ileten pipeline modülü. `CountingModule`/`RecognitionModule` event üretir (davranış-korumalı ek). Spec: `docs/superpowers/specs/2026-07-02-a1-reportmanager-rest-alarm-design.md`.

**Tech Stack:** Python 3.10+, stdlib `urllib.request` (yeni bağımlılık YOK), mevcut `pipeline.py`/`modules/` altyapısı, `secrets_util` (api_key şifresi), pytest.

## Global Constraints

- Kod yorumları ASCII-translit Türkçe (diakritiksiz); belgeler tam Türkçe. (Repo konvansiyonu)
- Davranış-koruma: `reporting:` tanımsız mevcut config ile tam suite geçer, çalışma zamanı davranışı birebir aynı kalır.
- HTTP asla üretici/pipeline thread'inde senkron çağrılmaz; `send()` yalnız kuyruğa yazar.
- Yeni bağımlılık eklenmez (`requests` YOK; stdlib `urllib.request`).
- İsimlendirme AiEye: logger'lar `aieye.*` (rename sonrası durum).
- Testler venv ile: `venv/Scripts/python.exe -m pytest` (Windows) / `venv/bin/python -m pytest`.

---

### Task 0: Hazırlık — merge doğrulama, dal, rename straggler süpürme

**Files:**
- Modify: `pipeline.py`, `modules/__init__.py`, `modules/counting.py` (yalnız `facezoom`→`aieye` metin ikamesi)

**Interfaces:** yok (VCS + hijyen).

**Ön koşul:** PR #1 (`feat/config-driven-pipeline`) ve PR #2 (`rename/facezoom-to-aieye`) main'e merge edilmiş OLMALI. Değilse DURUP kullanıcıya bildir.

- [ ] **Step 1: Merge durumunu doğrula ve dal aç**

```bash
cd /d/proje/FaceZoom
git fetch origin
git log origin/main --oneline -5
```
Beklenen: hem pipeline commit'leri (`feat(worker): process() iki faz...`) hem rename commit'i (`refactor: FaceZoom projesini AiEye...`) görünür. Görünmüyorsa DUR.

```bash
git checkout main && git pull
git checkout -b feat/a1-reporting
```

- [ ] **Step 2: Rename straggler'larını süpür**

PR #1 dosyaları rename'den önce yazıldığı için `facezoom` kalıntısı içerir:

```bash
grep -rniE 'facezoom' pipeline.py modules/ tests/ | grep -v '\.pyc'
```
Beklenen (en az): `pipeline.py:11` (logger), `modules/counting.py:12` (logger), `modules/__init__.py:1` (yorum).

```bash
sed -i 's/facezoom/aieye/g; s/FaceZoom/AiEye/g; s/FACEZOOM/AIEYE/g' \
  pipeline.py modules/__init__.py modules/counting.py \
  modules/overlay.py modules/recognition.py modules/zoom.py \
  tests/helpers_pipe.py tests/test_pipeline_loader.py tests/test_pipeline_build_run.py \
  tests/test_module_counting.py tests/test_module_overlay.py \
  tests/test_module_recognition.py tests/test_module_zoom.py \
  tests/test_worker_pipeline_integration.py
grep -rniE 'facezoom' pipeline.py modules/ tests/ | grep -v '\.pyc'
```
Beklenen: ikinci grep BOŞ.

- [ ] **Step 3: Tam suite + commit**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: tümü PASS (merge sonrası ~113+ test).

```bash
git add -A
git commit -m "chore: rename straggler supurme (pipeline/modules facezoom -> aieye)"
```

---

### Task 1: ReportManager çekirdeği — kuyruk, rate-limit, kurucu fonksiyonlar

**Files:**
- Create: `reporting.py`
- Test: `tests/test_reporting.py`

**Interfaces:**
- Consumes: `secrets_util.decrypt(token)`, `secrets_util.is_encrypted(token)` (mevcut).
- Produces (sonraki task'lar bunlara güvenir):
  - `class ReportManager` — `__init__(reporting_cfg: dict, api_key: str)` (thread BAŞLATMAZ);
    `send(event: dict) -> bool`; `can_send(key, now=None) -> bool`; `start() -> self`;
    `stop()`; iç alanlar: `_queue` (deque), `_post(payload) -> bool`, `_snapshot()`.
  - `build_report_manager(config: dict) -> ReportManager | None` — enabled değilse/gateway
    boşsa/api_key çözülemezse None; yoksa `start()` edilmiş instance.
  - `resolve_reporting(config: dict, camera_cfg: dict | None) -> dict` — sığ merge, kamera öncelikli.
  - Event dict sözleşmesi: `{"type", "camera", "ts", "direction"?, "name"?, "jpeg"?(bytes), "branch_id"?}`.
  - Payload dict (kuyrukta, JSON-uyumlu): `{"camera", "branchId", "eventType", "triggeredAt"(ISO-8601 UTC), "direction", "name", "message", "image"(base64|None)}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reporting.py
import time

import secrets_util
import reporting
from reporting import ReportManager, build_report_manager, resolve_reporting


def _rm(tmp_path, **over):
    cfg = {"gateway_base": "http://gw", "branch_id": "S1",
           "cooldown_seconds": 60, "once_per_day": False,
           "queue_path": str(tmp_path / "q.json"), "max_queue": 3}
    cfg.update(over)
    return ReportManager(cfg, api_key="KEY")


def test_build_none_when_disabled_or_empty(tmp_path):
    assert build_report_manager({}) is None
    assert build_report_manager({"reporting": {"enabled": False}}) is None
    assert build_report_manager(
        {"reporting": {"enabled": True, "gateway_base": ""}}) is None


def test_build_none_when_key_unresolved(tmp_path, monkeypatch):
    # sifreleme anahtari YOK -> enc$ cozulmez -> raporlama kapali
    monkeypatch.delenv("AIEYE_SECRET_KEY", raising=False)
    monkeypatch.delenv("AIEYE_KEY_FILE", raising=False)
    monkeypatch.setattr(secrets_util, "_cache", {"loaded": False, "fernet": None})
    cfg = {"reporting": {"enabled": True, "gateway_base": "http://gw",
                         "api_key": "enc$bozuktoken",
                         "queue_path": str(tmp_path / "q.json")}}
    assert build_report_manager(cfg) is None


def test_build_basarili_start_edilir(tmp_path):
    cfg = {"reporting": {"enabled": True, "gateway_base": "http://gw",
                         "api_key": "duzanahtar",
                         "queue_path": str(tmp_path / "q.json")}}
    rm = build_report_manager(cfg)
    assert rm is not None and rm._thread is not None
    rm.stop()


def test_gateway_bos_send_false(tmp_path):
    rm = _rm(tmp_path, gateway_base="")
    assert rm.send({"type": "t", "camera": "K", "ts": 1.0}) is False


def test_cooldown_ayni_anahtar_reddedilir(tmp_path):
    rm = _rm(tmp_path)
    e = {"type": "counting_crossing", "camera": "Kam", "ts": 1000.0}
    assert rm.send(e) is True
    assert rm.send({**e, "ts": 1030.0}) is False           # 60 sn dolmadi
    assert rm.send({**e, "ts": 1061.0}) is True            # doldu
    assert rm.send({**e, "camera": "Diger", "ts": 1030.0}) is True  # farkli anahtar


def test_once_per_day(tmp_path):
    rm = _rm(tmp_path, once_per_day=True, cooldown_seconds=0, max_queue=10)
    day1 = time.mktime((2026, 7, 2, 10, 0, 0, 0, 0, -1))
    assert rm.send({"type": "t", "camera": "K", "ts": day1}) is True
    assert rm.send({"type": "t", "camera": "K", "ts": day1 + 3600}) is False
    assert rm.send({"type": "t", "camera": "K", "ts": day1 + 86400}) is True


def test_fifo_drop_en_eski_duser(tmp_path):
    rm = _rm(tmp_path, cooldown_seconds=0, max_queue=3)
    for i in range(5):
        assert rm.send({"type": "t", "camera": f"K{i}", "ts": 1000.0 + i}) is True
    assert [p["camera"] for p in rm._queue] == ["K2", "K3", "K4"]


def test_resolve_reporting_kamera_oncelikli():
    cfg = {"reporting": {"branch_id": "G", "events": ["a"], "cooldown_seconds": 60}}
    cam = {"name": "K", "reporting": {"branch_id": "S2"}}
    m = resolve_reporting(cfg, cam)
    assert m["branch_id"] == "S2"
    assert m["events"] == ["a"]
    assert m["cooldown_seconds"] == 60
    assert resolve_reporting(cfg, None)["branch_id"] == "G"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_reporting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reporting'`

- [ ] **Step 3: Write minimal implementation**

```python
# reporting.py
# -----------------------------------------------------------------------------
# REST rapor/alarm katmani (Faz A1). ReportManager pipeline'dan HABERSIZDIR:
# olay al -> kuyrukla -> daemon thread REST'e gonderir. send() yalniz kuyruga
# yazar (rate-limit uygular); ag isi sicak yolda ASLA beklenmez.
# Cevrimdisi dayaniklilik: gonderim basarisizsa kuyruk queue_path'e snapshot
# edilir, aclista geri yuklenir. api_key secrets_util ile enc$ cozulur.
# -----------------------------------------------------------------------------

import base64
import collections
import json
import logging
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone

import secrets_util

log = logging.getLogger("aieye.reporting")

RETRY_DELAY = 5.0     # basarisiz POST sonrasi bekleme (sn)
HTTP_TIMEOUT = 10.0   # tek POST zaman asimi (sn)


def resolve_reporting(config, camera_cfg=None):
    """Global reporting + kamera-bazli override'i SIG birlestir (kamera oncelikli)."""
    merged = dict(config.get("reporting") or {})
    merged.update((camera_cfg or {}).get("reporting") or {})
    return merged


class ReportManager:
    """Olay kuyrugu + daemon gonderici. __init__ thread BASLATMAZ (testler icin);
    start() baslatir (build_report_manager cagirir)."""

    def __init__(self, reporting_cfg, api_key):
        self.gateway = (reporting_cfg.get("gateway_base") or "").rstrip("/")
        self.api_key = api_key or ""
        self.branch_id = reporting_cfg.get("branch_id") or ""
        self.cooldown = float(reporting_cfg.get("cooldown_seconds", 60))
        self.once_per_day = bool(reporting_cfg.get("once_per_day", False))
        self.queue_path = reporting_cfg.get("queue_path") or "report_queue.json"
        self._queue = collections.deque(maxlen=int(reporting_cfg.get("max_queue", 500)))
        self._lock = threading.Lock()
        # (camera, type) -> son gonderim epoch'u | "YYYY-MM-DD" (once_per_day)
        self._last_sent = {}
        self._stop_ev = threading.Event()
        self._thread = None
        self._drop_count = 0
        self._load_snapshot()

    # ---- rate limit ----
    def _day(self, now):
        return time.strftime("%Y-%m-%d", time.localtime(now))

    def can_send(self, key, now=None):
        """Durum DEGISTIRMEDEN cooldown/once_per_day kontrolu."""
        now = time.time() if now is None else now
        prev = self._last_sent.get(key)
        if prev is None:
            return True
        if self.once_per_day:
            return prev != self._day(now)
        return (now - prev) >= self.cooldown

    def send(self, event):
        """Olayi kuyruga al. Gateway yoksa / cooldown'daysa False."""
        if not self.gateway:
            return False
        key = (event.get("camera"), event.get("type"))
        now = event.get("ts") or time.time()
        if not self.can_send(key, now):
            log.debug("Rapor cooldown'da, atlandi: %s", key)
            return False
        payload = self._payload(event, now)
        with self._lock:
            if len(self._queue) == self._queue.maxlen:
                self._drop_count += 1
                if self._drop_count == 1 or self._drop_count % 50 == 0:
                    log.warning("Rapor kuyrugu dolu (%d): en eski dusuyor (#%d)",
                                self._queue.maxlen, self._drop_count)
            self._queue.append(payload)
        self._last_sent[key] = self._day(now) if self.once_per_day else now
        return True

    def _payload(self, event, now):
        """Event -> JSON-uyumlu payload (kuyrukta bu tutulur; snapshot kolay)."""
        img = event.get("jpeg")
        return {
            "camera": event.get("camera"),
            "branchId": event.get("branch_id") or self.branch_id,
            "eventType": event.get("type"),
            "triggeredAt": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "direction": event.get("direction"),
            "name": event.get("name"),
            "message": "%s @ %s" % (event.get("type"), event.get("camera")),
            "image": base64.b64encode(img).decode("ascii") if img else None,
        }

    # ---- gonderici thread (Task 2'de tamamlanir) ----
    def start(self):
        self._thread = threading.Thread(target=self._sender_loop,
                                        name="report-sender", daemon=True)
        self._thread.start()
        return self

    def _sender_loop(self):
        # Task 2'de dolduruluyor; simdilik bos dongu (stop calissin)
        while not self._stop_ev.is_set():
            self._stop_ev.wait(0.2)

    def _snapshot(self):
        pass   # Task 2

    def _load_snapshot(self):
        pass   # Task 2

    def stop(self):
        self._stop_ev.set()
        if self._thread is not None:
            self._thread.join(timeout=RETRY_DELAY + 1.0)


def build_report_manager(config):
    """reporting.enabled degilse None. api_key enc$ olup COZULEMIYORSA None
    (yanlis anahtarla dis API'ye istek ATILMAZ)."""
    rep = config.get("reporting") or {}
    if not rep.get("enabled", False):
        return None
    if not (rep.get("gateway_base") or "").strip():
        log.warning("reporting.enabled=true ama gateway_base bos -> raporlama KAPALI")
        return None
    raw = rep.get("api_key") or ""
    api_key = secrets_util.decrypt(raw)
    if secrets_util.is_encrypted(raw) and api_key == raw:
        log.warning("reporting.api_key cozulemedi (anahtar eksik/yanlis) "
                    "-> raporlama KAPALI")
        return None
    return ReportManager(rep, api_key).start()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_reporting.py -v`
Expected: PASS (8 test)

- [ ] **Step 5: Commit**

```bash
git add reporting.py tests/test_reporting.py
git commit -m "feat(reporting): ReportManager cekirdegi (kuyruk + rate-limit + kurucu)"
```

---

### Task 2: Gönderici thread — HTTP POST, offline snapshot, stop

**Files:**
- Modify: `reporting.py` (`_sender_loop`, `_post`, `_snapshot`, `_load_snapshot`, `stop`)
- Test: `tests/test_reporting.py` (ekleme)

**Interfaces:**
- Consumes: Task 1'in `ReportManager` iskeleti.
- Produces: `_post(payload) -> bool` (2xx=True); `_snapshot()` kuyruğu `queue_path`'e JSON yazar;
  `_load_snapshot()` açılışta dosyayı yükleyip SİLER; `stop()` thread'i kapatıp bekleyenleri diske yazar.

- [ ] **Step 1: Write the failing tests (mevcut dosyaya ekle)**

```python
# tests/test_reporting.py'ye EKLE
import base64
import json
from datetime import datetime, timezone


class _Resp:
    status = 200
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_post_dogru_istek_kurar(tmp_path, monkeypatch):
    seen = {}
    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["key"] = req.get_header("X-api-key")
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["timeout"] = timeout
        return _Resp()
    monkeypatch.setattr(reporting.urllib.request, "urlopen", fake_urlopen)
    rm = _rm(tmp_path)
    rm.send({"type": "counting_crossing", "camera": "Kam", "ts": 1000.0,
             "direction": "in", "jpeg": b"JJ"})
    assert rm._post(rm._queue[0]) is True
    assert seen["url"] == "http://gw/AiInput"
    assert seen["key"] == "KEY"
    b = seen["body"]
    assert b["eventType"] == "counting_crossing"
    assert b["direction"] == "in"
    assert b["branchId"] == "S1"
    assert b["triggeredAt"] == datetime.fromtimestamp(
        1000.0, tz=timezone.utc).isoformat()
    assert base64.b64decode(b["image"]) == b"JJ"


def test_post_hata_false(tmp_path, monkeypatch):
    def fail(req, timeout=None):
        raise OSError("baglanti yok")
    monkeypatch.setattr(reporting.urllib.request, "urlopen", fail)
    rm = _rm(tmp_path)
    rm.send({"type": "t", "camera": "K", "ts": 1.0})
    assert rm._post(rm._queue[0]) is False


def test_offline_snapshot_ve_reload(tmp_path):
    rm = _rm(tmp_path, cooldown_seconds=0)
    rm.send({"type": "t", "camera": "K1", "ts": 1.0})
    rm.send({"type": "t", "camera": "K2", "ts": 2.0})
    rm._snapshot()
    qp = tmp_path / "q.json"
    assert qp.exists()
    rm2 = _rm(tmp_path)     # ayni queue_path -> yukler + dosyayi siler
    assert [p["camera"] for p in rm2._queue] == ["K1", "K2"]
    assert not qp.exists()


def test_sender_loop_gonderir_ve_bosaltir(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp())
    rm = _rm(tmp_path, cooldown_seconds=0)
    rm.send({"type": "t", "camera": "K", "ts": 1.0})
    rm.start()
    deadline = time.time() + 3.0
    while rm._queue and time.time() < deadline:
        time.sleep(0.05)
    rm.stop()
    assert not rm._queue


def test_stop_bekleyenleri_diske_yazar(tmp_path):
    rm = _rm(tmp_path, cooldown_seconds=0)
    rm.send({"type": "t", "camera": "K", "ts": 1.0})
    rm.stop()                 # thread baslamadi ama kuyruk dolu -> snapshot
    assert (tmp_path / "q.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_reporting.py -v`
Expected: yeni 5 test FAIL (`_post` yok / snapshot boş no-op), Task 1 testleri PASS.

- [ ] **Step 3: Implement — reporting.py'de Task 1 iskeletini doldur**

`_sender_loop`, `_snapshot`, `_load_snapshot`, `stop` gövdelerini DEĞİŞTİR ve `_post` EKLE:

```python
    def _sender_loop(self):
        while not self._stop_ev.is_set():
            with self._lock:
                item = self._queue[0] if self._queue else None
            if item is None:
                self._stop_ev.wait(0.2)
                continue
            if self._post(item):
                with self._lock:
                    if self._queue and self._queue[0] is item:
                        self._queue.popleft()
            else:
                self._snapshot()             # cevrimdisi: birikeni diske yaz
                self._stop_ev.wait(RETRY_DELAY)

    def _post(self, payload):
        try:
            req = urllib.request.Request(
                self.gateway + "/AiInput",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "X-API-KEY": self.api_key},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                return 200 <= resp.status < 300
        except Exception as e:
            log.warning("Rapor gonderilemedi (%s): %s", self.gateway, e)
            return False

    def _snapshot(self):
        try:
            with self._lock:
                items = list(self._queue)
            with open(self.queue_path, "w", encoding="utf-8") as f:
                json.dump(items, f)
        except OSError:
            log.warning("Rapor kuyrugu diske yazilamadi: %s", self.queue_path)

    def _load_snapshot(self):
        if not os.path.exists(self.queue_path):
            return
        try:
            with open(self.queue_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            self._queue.extend(items)
            os.remove(self.queue_path)
            log.info("Bekleyen %d rapor diskten yuklendi", len(items))
        except (OSError, ValueError):
            log.warning("Rapor kuyrugu dosyasi okunamadi: %s", self.queue_path)

    def stop(self):
        """Kapanis: thread'i durdur, bekleyen raporlari diske yaz (kayip olmasin)."""
        self._stop_ev.set()
        if self._thread is not None:
            self._thread.join(timeout=RETRY_DELAY + 1.0)
        with self._lock:
            has_items = bool(self._queue)
        if has_items:
            self._snapshot()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_reporting.py -v`
Expected: PASS (13 test)

- [ ] **Step 5: Commit**

```bash
git add reporting.py tests/test_reporting.py
git commit -m "feat(reporting): daemon gonderici + HTTP POST + offline snapshot/reload"
```

---

### Task 3: ReportingModule (pipeline tüketicisi)

**Files:**
- Create: `modules/reporting.py`
- Test: `tests/test_module_reporting.py`

**Interfaces:**
- Consumes: `pipeline.PipelineModule`; `reporting.resolve_reporting`; `services["report_manager"]`
  (`send(event) -> bool` olan nesne veya None); `ctx["events"]` (Task 4 üretecek).
- Produces: `class ReportingModule(PipelineModule)` — `process(ctx)`: izinli tipleri filtreler,
  `crop`(ndarray)'ı JPEG bytes'a çevirir (`jpeg` alanı), kamera-merge `branch_id`'yi ekler, `send()` çağırır.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_module_reporting.py
import numpy as np

from modules.reporting import ReportingModule


class _FakeRM:
    def __init__(self):
        self.sent = []
    def send(self, ev):
        self.sent.append(ev)
        return True


def test_filtre_yalniz_izinli_tipler():
    rm = _FakeRM()
    m = ReportingModule()
    m.setup({"reporting": {"events": ["counting_crossing"]}}, "Kam",
            {"report_manager": rm})
    m.process({"events": [
        {"type": "counting_crossing", "camera": "Kam", "ts": 1.0},
        {"type": "capture_finished", "camera": "Kam", "ts": 1.0},
    ]})
    assert len(rm.sent) == 1
    assert rm.sent[0]["type"] == "counting_crossing"


def test_rm_yoksa_noop():
    m = ReportingModule()
    m.setup({}, "Kam", {"report_manager": None})
    m.process({"events": [{"type": "counting_crossing"}]})   # patlamamali


def test_capture_crop_jpege_cevrilir():
    rm = _FakeRM()
    m = ReportingModule()
    m.setup({}, "Kam", {"report_manager": rm})
    crop = np.zeros((8, 8, 3), np.uint8)
    ev_in = {"type": "capture_finished", "camera": "Kam", "ts": 1.0,
             "crop": crop, "bbox": (0, 0, 8, 8), "quality": 0.5}
    ctx = {"events": [ev_in]}
    m.process(ctx)
    sent = rm.sent[0]
    assert "crop" not in sent and isinstance(sent["jpeg"], bytes)
    assert "crop" in ctx["events"][0]   # orijinal event DEGISMEDI (kopya gonderildi)


def test_kamera_branch_override_eklenir():
    rm = _FakeRM()
    m = ReportingModule()
    cfg = {"reporting": {"branch_id": "G"},
           "cameras": [{"name": "Kam", "reporting": {"branch_id": "S2"}}]}
    m.setup(cfg, "Kam", {"report_manager": rm})
    m.process({"events": [{"type": "counting_crossing", "camera": "Kam", "ts": 1.0}]})
    assert rm.sent[0]["branch_id"] == "S2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_module_reporting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modules.reporting'`

- [ ] **Step 3: Write minimal implementation**

```python
# modules/reporting.py
# Rapor tuketici modulu: ctx['events'] icindeki izinli olaylari ReportManager'a
# iletir. Zincirde uretici modullerden (counting/recognition) SONRA yer almali.
# report_manager yoksa no-op. crop (ndarray) burada JPEG'e cevrilir; boylece
# ReportManager saf JSON-uyumlu veriyle calisir.

import cv2 as cv

from pipeline import PipelineModule
from reporting import resolve_reporting

DEFAULT_EVENTS = ("counting_crossing", "capture_finished")


class ReportingModule(PipelineModule):
    def setup(self, config, camera, services):
        self.rm = services.get("report_manager")
        cam_cfg = None
        for c in config.get("cameras") or []:
            if c.get("name") == camera:
                cam_cfg = c
                break
        rep = resolve_reporting(config, cam_cfg)
        self.event_types = set(rep.get("events") or DEFAULT_EVENTS)
        self.branch_id = rep.get("branch_id") or ""

    def process(self, ctx):
        if self.rm is None:
            return
        for ev in ctx.get("events") or []:
            if ev.get("type") not in self.event_types:
                continue
            ev = dict(ev)   # orijinali degistirme (baska tuketiciler okuyabilir)
            if self.branch_id:
                ev["branch_id"] = self.branch_id
            crop = ev.pop("crop", None)
            if crop is not None and ev.get("jpeg") is None:
                ok, buf = cv.imencode(".jpg", crop, [cv.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    ev["jpeg"] = buf.tobytes()
            self.rm.send(ev)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_module_reporting.py -v`
Expected: PASS (4 test)

- [ ] **Step 5: Commit**

```bash
git add modules/reporting.py tests/test_module_reporting.py
git commit -m "feat(modules): ReportingModule (ctx.events -> ReportManager)"
```

---

### Task 4: Üretici modüller — ctx["events"] (davranış-korumalı ek)

**Files:**
- Modify: `modules/counting.py` (process sonu), `modules/recognition.py` (process)
- Test: `tests/test_module_counting.py`, `tests/test_module_recognition.py` (ekleme)

**Interfaces:**
- Produces: `ctx["events"]` listesine:
  - CountingModule: `{"type": "counting_crossing", "camera", "ts", "direction", "name": None, "jpeg"}`
  - RecognitionModule: `{"type": "capture_finished", "camera", "ts", "crop", "bbox", "quality"}`
- KURAL: mevcut `counting_store.record` / `on_capture` çağrıları AYNEN kalır; event yalnız EKTİR.
  RecognitionModule event'i `on_capture=None` olsa da üretir (CLI'da da rapor çalışsın).

- [ ] **Step 1: Write the failing tests (mevcut dosyalara ekle)**

```python
# tests/test_module_counting.py'ye EKLE
def test_counting_event_uretir():
    lc = LineCrossingCounter(line=(0.0, 0.5, 1.0, 0.5))
    store = CountingStore()
    m = CountingModule()
    m.setup({}, "Kam", {"line_counter": lc, "counting_store": store,
                        "name_resolver": None, "manager": _FakeMgr()})
    c1 = _ctx([(1, (90, 40, 20, 20))])
    m.process(c1)
    assert not c1.get("events")            # gecis yok -> event yok
    c2 = _ctx([(1, (90, 140, 20, 20))])
    m.process(c2)
    evs = c2.get("events") or []
    assert len(evs) == 1
    ev = evs[0]
    assert ev["type"] == "counting_crossing"
    assert ev["camera"] == "Kam"
    assert ev["direction"] in ("in", "out")
    assert ev["name"] is None
```

```python
# tests/test_module_recognition.py'ye EKLE
def test_recognition_event_uretir_on_capture_olmadan_da():
    m = RecognitionModule()
    m.setup({}, "Kam", {"on_capture": None})
    ctx = {"finished": [_Track(1)], "camera": "Kam", "now": 5.0}
    m.process(ctx)
    evs = ctx.get("events") or []
    assert len(evs) == 1
    ev = evs[0]
    assert ev["type"] == "capture_finished"
    assert ev["crop"] == "crop1"
    assert ev["quality"] == 0.9
    assert ev["ts"] == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_module_counting.py tests/test_module_recognition.py -v`
Expected: yeni 2 test FAIL (`events` üretilmiyor), mevcut testler PASS.

- [ ] **Step 3: Implement — modules/counting.py**

`process()` döngüsünde, `eid = self.counting_store.record(...)` satırından ÖNCE ekle
(`for tid, direction in events:` bloğunun içinde, `jpeg` hesaplandıktan sonra):

```python
            # Rapor katmanina blackboard olayi (ReportingModule tuketir).
            # Isim asenkron cozulur -> ilk raporda None gider (YAGNI, spec §2).
            ctx.setdefault("events", []).append({
                "type": "counting_crossing", "camera": ctx.get("camera"),
                "ts": now, "direction": direction, "name": None, "jpeg": jpeg,
            })
```

- [ ] **Step 4: Implement — modules/recognition.py**

`process()` metodunu şu halde DEĞİŞTİR (event üretimi `on_capture`'dan bağımsız):

```python
    def process(self, ctx):
        finished = ctx.get("finished") or []
        for tr in finished:
            # Rapor katmanina blackboard olayi (ReportingModule tuketir);
            # on_capture olmasa da uretilir (CLI'da da rapor calissin).
            ctx.setdefault("events", []).append({
                "type": "capture_finished", "camera": ctx.get("camera"),
                "ts": ctx.get("now"), "crop": tr.best_crop,
                "bbox": tr.best_bbox, "quality": tr.best_score,
            })
        if self.on_capture is None:
            return
        for tr in finished:
            self.on_capture(ctx.get("camera"), tr.best_crop, tr.best_bbox,
                            tr.best_score, tr.first_seen, tr.last_seen, tr.best_time)
```

- [ ] **Step 5: Run FULL suite (davranış-koruma kanıtı)**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: TÜMÜ PASS (mevcut assertion'lar dahil).

- [ ] **Step 6: Commit**

```bash
git add modules/counting.py modules/recognition.py tests/test_module_counting.py tests/test_module_recognition.py
git commit -m "feat(modules): counting/recognition ctx.events uretimi (davranis-korumali)"
```

---

### Task 5: Varsayılan zincir + CameraWorker servisi

**Files:**
- Modify: `pipeline.py` (`_default_chain`), `worker.py` (`CameraWorker.__init__` imza + services)
- Test: `tests/test_pipeline_build_run.py`, `tests/test_worker_pipeline_integration.py` (ekleme)

**Interfaces:**
- Consumes: Task 3'ün `modules.reporting:ReportingModule`'ü.
- Produces: `CameraWorker(..., report_manager=None)` kwarg'ı; `services["report_manager"]`;
  `_default_chain` `reporting.enabled` ise sona `"modules.reporting:ReportingModule"` ekler.

- [ ] **Step 1: Write the failing tests (mevcut dosyalara ekle)**

```python
# tests/test_pipeline_build_run.py'ye EKLE
def test_build_reporting_enabled_zincire_eklenir():
    cfg = {"zoom_enabled": False, "debug_overlay": False,
           "reporting": {"enabled": True}}
    p = pipeline.build_pipeline(cfg, "Kam", {}, is_counting=False)
    names = [type(m).__name__ for m in p.modules]
    assert names == ["RecognitionModule", "ReportingModule"]


def test_build_reporting_kapali_zincire_girmez():
    cfg = {"zoom_enabled": False, "debug_overlay": False}
    p = pipeline.build_pipeline(cfg, "Kam", {}, is_counting=False)
    names = [type(m).__name__ for m in p.modules]
    assert "ReportingModule" not in names
```

```python
# tests/test_worker_pipeline_integration.py'ye EKLE
def test_report_manager_service_ve_modul(monkeypatch):
    class _Cam:
        name = "Kam"
        connected = True
        def read_detect(self):
            return None, 0
        def read_hires(self):
            return np.zeros((120, 160, 3), np.uint8), 1

    class _RM:
        def send(self, ev):
            return True

    cfg = {"detector_backend": "mediapipe", "zoom_enabled": False,
           "debug_overlay": False, "output_size": [160, 120],
           "detect_interval": 1, "reporting": {"enabled": True}}
    w = W.CameraWorker(_Cam(), cfg, report_manager=_RM())
    names = [type(m).__name__ for m in w._pipeline.modules]
    assert "ReportingModule" in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_pipeline_build_run.py tests/test_worker_pipeline_integration.py -v`
Expected: yeni testler FAIL (`ReportingModule` zincirde yok / `report_manager` kwarg yok).

- [ ] **Step 3: Implement — pipeline.py `_default_chain`**

`if config.get("debug_overlay", True):` bloğundan SONRA, `return chain`'den önce ekle:

```python
    # Raporlama acik ise zincirin SONUNA ekle: process fazi liste sirasiyla
    # calisir -> uretici moduller (recognition/counting) event'leri once yazar.
    if (config.get("reporting") or {}).get("enabled", False):
        chain.append("modules.reporting:ReportingModule")
```

- [ ] **Step 4: Implement — worker.py**

İmzaya kwarg ekle:

```python
    def __init__(self, camera, config, db=None, on_capture=None,
                 line_counter=None, counting_store=None, name_provider=None,
                 report_manager=None):
```

`self.name_provider = name_provider` satırından sonra ekle:

```python
        # REST rapor/alarm kuyrugu (uygulama basina TEK ornek; None = kapali)
        self.report_manager = report_manager
```

`services` sözlüğüne (`"manager": self.manager,` satırından sonra) ekle:

```python
            "report_manager": self.report_manager,
```

- [ ] **Step 5: Run tests + full suite**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: TÜMÜ PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline.py worker.py tests/test_pipeline_build_run.py tests/test_worker_pipeline_integration.py
git commit -m "feat(pipeline+worker): ReportingModule varsayilan zincirde + report_manager servisi"
```

---

### Task 6: Uygulama bağlama (webui/live/main) + config şablonu

**Files:**
- Modify: `live.py` (`_PreviewWorker.__init__`, `LiveManager.__init__`, `LiveManager.ensure`)
- Modify: `webui.py` (singleton kurulum bölgesi, `LiveManager(...)` çağrısı)
- Modify: `main.py` (kurulum + `finally`)
- Modify: `config.example.yaml` (reporting bölümü)

**Interfaces:**
- Consumes: `reporting.build_report_manager(config)`, `ReportManager.stop()`.
- Produces: uygulama başına TEK `ReportManager`; her `CameraWorker`'a `report_manager=` ile taşınır.

- [ ] **Step 1: live.py — parametre taşıma**

`_PreviewWorker.__init__` imzasına `report_manager=None` ekle ve `CameraWorker` çağrısına geçir:

```python
    def __init__(self, camera, config, db=None, on_capture=None,
                 line_counter=None, counting_store=None, name_provider=None,
                 report_manager=None):
        self.camera = camera
        self.worker = CameraWorker(camera, config, db=db, on_capture=on_capture,
                                   line_counter=line_counter,
                                   counting_store=counting_store,
                                   name_provider=name_provider,
                                   report_manager=report_manager)
```

`LiveManager.__init__` imzasına `report_manager=None` ekle; `self.name_provider = ...`
satırından sonra `self.report_manager = report_manager` ekle. `ensure()` içindeki
`_PreviewWorker(...)` çağrısına `report_manager=self.report_manager,` parametresi ekle
(raporlama sayım kamerasına özgü DEĞİLDİR; her kameraya verilir):

```python
            pw = _PreviewWorker(cam, self.config, db=self.db,
                                on_capture=self.on_capture,
                                line_counter=lc, counting_store=cs,
                                name_provider=npv,
                                report_manager=self.report_manager).start()
```

- [ ] **Step 2: webui.py — kurulum**

Sayım bloğunun bitiminden sonra (`_name_provider` tanımından önce) ekle:

```python
# --- REST raporlama/alarm (opsiyonel; reporting.enabled ile acilir) ---
from reporting import build_report_manager

REPORTER = build_report_manager(CONFIG)
if REPORTER is not None:
    import atexit
    atexit.register(REPORTER.stop)   # kapaniss: bekleyen raporlar diske
    log.info("REST raporlama AKTIF: %s",
             (CONFIG.get("reporting") or {}).get("gateway_base"))
```

`LIVE = LiveManager(...)` çağrısına `report_manager=REPORTER` ekle:

```python
LIVE = LiveManager(CONFIG, db=None, on_capture=_on_capture,
                   counting_camera=_COUNT_CAM, line_counter=_LINE_COUNTER,
                   counting_store=COUNTING, name_provider=_name_provider,
                   report_manager=REPORTER)
```

- [ ] **Step 3: main.py — kurulum + kapanış**

`from worker import CameraWorker` satırından sonra ekle: `from reporting import build_report_manager`

`cameras = build_cameras(config)` satırından ÖNCE ekle:

```python
    report_manager = build_report_manager(config)
```

Worker kurulum satırını DEĞİŞTİR:

```python
    workers = [CameraWorker(cam, config, db, report_manager=report_manager)
               for cam in cameras]
```

`finally:` bloğunda `db.close()` satırından ÖNCE ekle:

```python
        if report_manager is not None:
            report_manager.stop()    # bekleyen raporlar diske (kayip olmasin)
```

- [ ] **Step 4: config.example.yaml — reporting bölümü**

`# ---- islem hatti (pipeline) ----` bölümünden SONRA ekle:

```yaml
# ---- raporlama / alarm (REST) ----
# Opsiyonel. enabled=false ya da bolum tanimsizsa SIFIR etki (thread acilmaz).
# Olaylar (sayim gecisi, biten gorunum) arka planda POST {gateway_base}/AiInput
# adresine gonderilir (header X-API-KEY). Ag koptugunda kuyruk report_queue.json'a
# yazilir, geri gelince gonderilir. api_key 'enc$...' olarak sifreli tutulabilir
# (secrets_util; AIEYE_SECRET_KEY anahtariyla cozulur).
reporting:
  enabled: false
  gateway_base: ""            # or. "https://ornek-gateway.example.com"
  api_key: ""                 # enc$... (onerilen) veya duz metin
  branch_id: ""
  cooldown_seconds: 60        # (kamera, olay-tipi) basina en az bu aralikla
  once_per_day: false
  events: ["counting_crossing", "capture_finished"]
  queue_path: "report_queue.json"
  max_queue: 500
# Kamera-bazli override (kamera blogunda): reporting: { branch_id: "SUBE-2" }
```

- [ ] **Step 5: Full suite + import smoke**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: TÜMÜ PASS.

Run: `venv/Scripts/python.exe -c "import reporting, main; from modules.reporting import ReportingModule; print('OK')"`
Expected: `OK` (import hatası yok; webui import'u Flask app kurduğu için smoke'a dahil edilmez).

- [ ] **Step 6: Commit**

```bash
git add live.py webui.py main.py config.example.yaml
git commit -m "feat(app): ReportManager kurulum/tasima (webui+main) + config sablonu"
```

---

### Task 7: Uçtan uca doğrulama + PR

**Files:** yok (doğrulama + VCS).

- [ ] **Step 1: Kabul kriteri 2-3 için yerel sahte gateway ile e2e smoke**

Geçici script `scratch_e2e_reporting.py` (commit ETME, sonra sil):

```python
# scratch_e2e_reporting.py — yerel sahte gateway'e gercek ReportManager gonderimi
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from reporting import build_report_manager

hits = []

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        hits.append((self.path, self.headers.get("X-API-KEY"),
                     json.loads(self.rfile.read(n))))
        self.send_response(200)
        self.end_headers()
    def log_message(self, *a):
        pass

srv = HTTPServer(("127.0.0.1", 18099), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()

cfg = {"reporting": {"enabled": True, "gateway_base": "http://127.0.0.1:18099",
                     "api_key": "TESTKEY", "branch_id": "S1",
                     "cooldown_seconds": 0,
                     "queue_path": "scratch_q.json"}}
rm = build_report_manager(cfg)
rm.send({"type": "counting_crossing", "camera": "Kam", "ts": time.time(),
         "direction": "in", "jpeg": b"XX"})
rm.send({"type": "capture_finished", "camera": "Kam", "ts": time.time()})
time.sleep(1.5)
rm.stop()
srv.shutdown()
assert len(hits) == 2, hits
assert all(p == "/AiInput" and k == "TESTKEY" for p, k, _ in hits)
print("E2E OK:", [b["eventType"] for _, _, b in hits])
```

Run: `venv/Scripts/python.exe scratch_e2e_reporting.py`
Expected: `E2E OK: ['counting_crossing', 'capture_finished']`

```bash
rm -f scratch_e2e_reporting.py scratch_q.json
```

- [ ] **Step 2: Tam suite son kontrol**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: TÜMÜ PASS.

- [ ] **Step 3: Spec+plan dokümanlarını dala al ve PR aç**

`docs/a1-reportmanager-spec` dalındaki iki dokümanı bu dala getir:

```bash
git checkout docs/a1-reportmanager-spec -- docs/superpowers/specs/2026-07-02-a1-reportmanager-rest-alarm-design.md docs/superpowers/plans/2026-07-02-a1-reportmanager-rest-alarm.md
git add docs/superpowers/
git commit -m "docs(a1): ReportManager spec + uygulama plani"
git push -u origin feat/a1-reporting
```

PR: `gh` + `GH_TOKEN` yöntemiyle (bkz. memory `gh-cli-pr-workflow`), base `main`,
başlık: `feat: REST rapor/alarm katmani (ReportManager, Faz A1)`.

---

## Self-Review (plan yazarı)

- **Spec kapsaması:** §3 mimari → Task 1-3; §4 blackboard → Task 4; §5 ReportManager → Task 1-2;
  §6 modül → Task 3; §7 config → Task 6 Step 4; §8 bağlama → Task 5-6; §9 hata → Task 1-2
  (build None / cooldown / FIFO / api_key) + mevcut `Pipeline._run_one` (değişiklik gerekmez);
  §10 test planı → Task 1-5 test adımları (spec test 1-8 + modül testleri birebir);
  §11 kabul → Task 4 Step 5 + Task 6 Step 5 (kriter 1), Task 7 Step 1 (kriter 2-3),
  api_key enc$ (kriter 4) Task 1 test + Task 6 config yorumu. Boşluk yok.
- **Tip tutarlılığı:** `send(event)->bool`, event alanları (`type/camera/ts/direction/name/jpeg/branch_id`),
  payload alanları (`camera/branchId/eventType/triggeredAt/direction/name/message/image`),
  `resolve_reporting(config, camera_cfg)` — Task 1'de tanımlandı, Task 2/3/7 aynı adlarla kullanıyor. ✓
- **Anchor doğruluğu:** worker.py imza/services (satır 104-105, 196-202), live.py imzalar
  (36-42, 98-108, 156-159), webui.py kurulum bölgesi (113-163), main.py finally (129-138),
  pipeline._default_chain (107-115) — hepsi feat dalındaki mevcut koddan birebir alındı.
  Task 0'daki rename süpürmesi yalnız `facezoom`→`aieye` metnini değiştirir, anchor yapılarını bozmaz. ✓
- **Placeholder taraması:** TBD/TODO yok; her adımda tam kod/komut var. ✓
