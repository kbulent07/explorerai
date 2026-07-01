# live.py
# -----------------------------------------------------------------------------
# Web arayuzu icin CANLI onizleme yoneticisi.
#
# Her kamera icin (ada gore) arka planda tek bir isleyici thread calisir:
# hires akisi okur, yuz algilar, FrameTransformer ile zoom yapar ve son kareyi
# JPEG olarak saklar. Tarayicidaki <img> etiketleri bu JPEG'i MJPEG akisi olarak
# alir. Birden cok izleyici ayni kareyi paylasir (kamera basina TEK baglanti).
#
# DB'ye YAZMAZ -> salt onizleme. Kayit/best-shot isi main.py'nin gorevidir.
# Kameralar ada gore tutuldugu icin, web ayarlar ekranindan yeni eklenen bir
# kamera, uygulama yeniden baslatilmadan izlenebilir.
# -----------------------------------------------------------------------------

import logging
import threading
import time

import cv2 as cv

import config_store
from camera import Camera
from worker import CameraWorker

log = logging.getLogger("facezoom.live")


class _PreviewWorker:
    """Tek kamera icin isleyici thread'i.

    Son zoomlu kareyi JPEG olarak tutar (canli onizleme) VE db verilmisse
    en-net kareleri galeriye kaydeder. Boylece tek worker hem yayini besler
    hem yakalama yapar -> ayrica main.py calistirmaya gerek kalmaz.
    """

    def __init__(self, camera, config, db=None, on_capture=None,
                 line_counter=None, counting_store=None, name_provider=None):
        self.camera = camera
        self.worker = CameraWorker(camera, config, db=db, on_capture=on_capture,
                                   line_counter=line_counter,
                                   counting_store=counting_store,
                                   name_provider=name_provider)
        # Onizleme/isleme hizi (CPU): MJPEG icin ~12 fps yeterli, 25'e gerek yok.
        fps = max(1, int(config.get("preview_fps", 12)))
        self._frame_interval = 1.0 / fps
        self.jpeg = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.camera.start()
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self.camera.stop()

    def _loop(self):
        quality = [cv.IMWRITE_JPEG_QUALITY, 80]
        while not self._stop.is_set():
            t0 = time.time()
            try:
                out = self.worker.process()
            except Exception as e:
                log.exception("[%s] onizleme isleme hatasi: %s", self.camera.name, e)
                out = None
            if out is not None:
                ok, buf = cv.imencode(".jpg", out, quality)
                if ok:
                    with self._lock:
                        self.jpeg = buf.tobytes()
            # Hedef fps'i koru: isleme suresini dusserek bekle (CPU tasarrufu)
            dt = time.time() - t0
            self._stop.wait(max(0.0, self._frame_interval - dt))

    def get_jpeg(self):
        with self._lock:
            return self.jpeg


class LiveManager:
    """Kamera adina gore onizleme thread'lerini tembel (lazy) baslatir."""

    def __init__(self, config, db=None, on_capture=None,
                 counting_camera=None, line_counter=None, counting_store=None,
                 name_provider=None):
        self.config = config
        self.db = db                # verilirse worker'lar diske/DB yakalar
        self.on_capture = on_capture  # verilirse her yakalamada cagrilir (bellek deposu)
        # GIRIS/CIKIS sayimi: yalniz bu ada sahip kameranin worker'i sayaca baglanir.
        self.counting_camera = counting_camera
        self.line_counter = line_counter
        self.counting_store = counting_store
        self.name_provider = name_provider   # gecis aninda yuz -> isim (opsiyonel)
        self._workers = {}          # name -> _PreviewWorker
        self._lock = threading.Lock()
        # None = config varsayilani; True/False = tum kameralar icin zorlanmiss
        # zoom durumu (sonradan baslayan worker'lar da bunu alir).
        self._zoom_override = None
        # Elle "baglantisi kesilmiss" kameralar: ensure() bunlari OTOMATIK
        # baslatmaz (RTSP tamamen kapali kalir) -> kullanici "Baglan" diyene kadar.
        self._disconnected = set()

    def _camera_cfg(self, name):
        """Guncel config.yaml'dan ada gore kamera ayarini bul."""
        for _idx, cam in config_store.list_cameras():
            if cam.get("name") == name:
                return cam
        # Webcam test modu (config_store kamera dondurmezse build_cameras'taki ad)
        if self.config.get("webcam_test_index") is not None and name == "Webcam":
            return {"name": "Webcam", "_webcam": True}
        return None

    def ensure(self, name):
        """Verilen kamerayi (gerekirse) baslat, _PreviewWorker dondur veya None."""
        with self._lock:
            if name in self._workers:
                return self._workers[name]
            # Elle baglantisi kesildiyse OTOMATIK baslatma (kamera tamamen bos)
            if name in self._disconnected:
                return None

            cfg = self._camera_cfg(name)
            if cfg is None:
                return None

            if cfg.get("_webcam"):
                cam = Camera(name="Webcam",
                             detect_source=int(self.config["webcam_test_index"]),
                             is_webcam=True)
            else:
                cam = Camera(
                    name=cfg["name"],
                    detect_source=cfg["detect_url"],
                    hires_source=cfg.get("hires_url"),
                )
            # Yalniz sayim kamerasina cizgi-gecis sayaci + depo + isim saglayici bagla
            is_count_cam = (name == self.counting_camera)
            lc = self.line_counter if is_count_cam else None
            cs = self.counting_store if is_count_cam else None
            npv = self.name_provider if is_count_cam else None
            pw = _PreviewWorker(cam, self.config, db=self.db,
                                on_capture=self.on_capture,
                                line_counter=lc, counting_store=cs,
                                name_provider=npv).start()
            if self._zoom_override is not None:
                pw.worker.set_zoom(self._zoom_override)   # global zoom durumunu uygula
            self._workers[name] = pw
            log.info("Kamera isleyici baslatildi: %s", name)
            return pw

    def start_all(self):
        """Yapilandirilmiss tum kameralari baslat (surekli yakalama icin)."""
        for name in self.available_names():
            self.ensure(name)

    def get_jpeg(self, name):
        pw = self.ensure(name)
        return pw.get_jpeg() if pw else None

    def available_names(self):
        """Izlenebilir kamera adlari (config + webcam test)."""
        names = [cam.get("name") for _i, cam in config_store.list_cameras()]
        if self.config.get("webcam_test_index") is not None:
            names.append("Webcam")
        return names

    def set_zoom(self, name, enabled):
        """Tek kameranin canli pan-zoom'unu ac/kapat (calissirken)."""
        with self._lock:
            pw = self._workers.get(name)
        if pw is not None:
            pw.worker.set_zoom(enabled)
            return True
        return False

    def set_zoom_all(self, enabled):
        """Tum kameralar icin zoom'u ac/kapat + sonraki worker'lar icin hatirla."""
        with self._lock:
            self._zoom_override = bool(enabled)
            workers = list(self._workers.values())
        for pw in workers:
            pw.worker.set_zoom(enabled)

    def stop_camera(self, name):
        """Bir kameranin isleyici thread'ini durdur ve kaydini sil.

        Kamera config'ten silinince cagrilmali; aksi halde silinen kameranin
        worker'i RTSP okumaya + yakalamaya DEVAM eder. Thread join'i kilit
        DISINDA yapilir (kamera.stop() okuyucu thread'leri bekler -> yavass olabilir).
        """
        with self._lock:
            pw = self._workers.pop(name, None)
        if pw is not None:
            pw.stop()
            log.info("Kamera isleyici durduruldu: %s", name)
        return pw is not None

    def disconnect(self, name):
        """Kamerayla baglantiyi TAMAMEN kes: worker thread'i durdur, RTSP akisini
        birak ve 'kesildi' isaretle ki ensure() onu otomatik baslatmasin."""
        with self._lock:
            self._disconnected.add(name)
        return self.stop_camera(name)

    def reconnect(self, name):
        """'Kesildi' isaretini kaldir ve kamerayi yeniden baslat."""
        with self._lock:
            self._disconnected.discard(name)
        return self.ensure(name) is not None

    def is_disconnected(self, name):
        with self._lock:
            return name in self._disconnected

    def disconnected_names(self):
        with self._lock:
            return set(self._disconnected)

    def stop_all(self):
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for pw in workers:
            pw.stop()
