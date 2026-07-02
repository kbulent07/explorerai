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
        # _last_sent kilitsiz: anahtar (camera, type) kamera-thread'ine ozgu oldugundan
        # es-zamanli yazim ayni anahtara gelmez (GIL tekil dict islemini atomik yapar).
        # Paylasilan-anahtar cagiran eklenirse kilit altina alinmali.
        # (camera, type) -> son gonderim epoch'u | "YYYY-MM-DD" (once_per_day)
        self._last_sent = {}
        self._stop_ev = threading.Event()
        self._thread = None
        self._drop_count = 0
        self._dirty = False  # kuyruk son snapshot'tan beri degisti mi
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
            self._dirty = True
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
                        self._dirty = True
                    now_empty = not self._queue
                if now_empty:
                    self._remove_snapshot()  # hepsi gitti -> bayat kopya kalmasin
            else:
                if self._dirty:
                    self._snapshot()         # cevrimdisi: birikeni diske yaz
                    self._dirty = False
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
            tmp_path = self.queue_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(items, f)
            os.replace(tmp_path, self.queue_path)  # atomik: yari-yazik dosya kalmaz
        except OSError:
            log.warning("Rapor kuyrugu diske yazilamadi: %s", self.queue_path)

    def _remove_snapshot(self):
        # gonderilmis kuyrugun bayat kopyasi diske kalmasin (mukerrer gonderim onlenir)
        try:
            os.remove(self.queue_path)
        except OSError:
            pass

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
            # bozuk dosyayi kenara al: her aclista ayni uyari donmesin
            try:
                os.replace(self.queue_path, self.queue_path + ".bad")
            except OSError:
                pass

    def stop(self):
        """Kapanis: thread'i durdur, bekleyen raporlari diske yaz (kayip olmasin)."""
        self._stop_ev.set()
        if self._thread is not None:
            self._thread.join(timeout=RETRY_DELAY + 1.0)
        with self._lock:
            has_items = bool(self._queue)
        if has_items:
            self._snapshot()
        else:
            self._remove_snapshot()


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
