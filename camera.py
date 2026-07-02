# camera.py
# -----------------------------------------------------------------------------
# Thread'li RTSP okuyucu + otomatik yeniden baglanma.
#
# Neden ayri thread? cv2.VideoCapture dahili bir buffer tutar; tek thread'de
# yavass islersek bu buffer dolar ve gecmiss kareler birikir (gecikme + takilma).
# Her akisi ayri thread'de surekli okuyup yalniz EN GUNCEL kareyi saklayarak
# her zaman "su an" goruntusuyle calisiriz.
#
# Bir kameranin detect (sub-stream) ve hires (main-stream) akislari AYRI birer
# StreamReader'dir; senkron sart degildir -> algilama anindaki en guncel hires
# kare best-shot icin kullanilir.
# -----------------------------------------------------------------------------

import logging
import os
import threading
import time

import cv2 as cv

log = logging.getLogger("aieye.camera")

# Hikvision UDP'de bozuk kare/paket kaybi yapabilir; TCP zorla.
# stimeout (mikrosaniye): akis DONARSA (freeze / sessiz kopma) read() sonsuza
# kadar bloke olmasin -> 5 sn icinde hata doner, otomatik yeniden baglanma tetiklenir.
# VideoCapture acilmadan ONCE ayarlanmali.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|stimeout;5000000")


class StreamReader:
    """Tek bir video kaynagini (RTSP URL veya webcam index) thread'de okur.

    En guncel kareyi self ile paylasir; read() her zaman son kareyi dondurur.
    Baglanti koparsa backoff'lu olarak yeniden baglanmayi dener; cagiran tarafin
    cokmesine yol acmaz.
    """

    def __init__(self, source, name="stream", is_webcam=False,
                 reconnect_min=1.0, reconnect_max=30.0):
        self.source = source
        self.name = name
        self.is_webcam = is_webcam
        self.reconnect_min = reconnect_min
        self.reconnect_max = reconnect_max

        self._cap = None
        self._frame = None
        self._frame_id = 0          # her yeni karede artar
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._connected = False
        self._thread = None

    # ---- yasam dongusu ------------------------------------------------------

    def start(self):
        self._thread = threading.Thread(target=self._run, name=f"reader-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            # stimeout=5 sn: donmuss akista read() 5 sn'ye kadar bloke olabilir.
            # join'i bunun USTUNDE bekle ki okuyucu thread normalde kendi cap'ini
            # birakip bitsin (asagidaki release ancak thread OLMEDIYSE calisir).
            self._thread.join(timeout=6.0)
        # cv2.VideoCapture THREAD-SAFE DEGIL: cap'i yalniz okuyucu thread artik
        # cap.read() icinde OLMADIGINDA (thread bitti) birak. Aksi halde stop()
        # ile _run() ayni cap'i eszamanli kullanip cokerdi. _run zaten cikarken
        # kendi cap'ini birakir; bu yalniz thread hic baslamadi/takildi yedegi.
        if (self._thread is None or not self._thread.is_alive()) and self._cap is not None:
            self._cap.release()

    # ---- okuma --------------------------------------------------------------

    def read(self):
        """(frame, frame_id) dondurur. Henuz kare yoksa (None, 0)."""
        with self._lock:
            if self._frame is None:
                return None, 0
            return self._frame, self._frame_id

    @property
    def connected(self):
        return self._connected

    # ---- ic isleyiss --------------------------------------------------------

    def _open(self):
        if self.is_webcam:
            cap = cv.VideoCapture(self.source)
        else:
            # Parola config'te sifreli (enc$...) olabilir -> VideoCapture'a GERCEK
            # URL verilir. Duz-metin URL'ler oldugu gibi gecer (geri uyum).
            import secrets_util
            # Sifre COZULEMIYORSA (anahtar eksik/yanlis) enc$ ham parola gibi gider
            # ve kimlik dogrulama basarisiz olur -> sebebi NET logla (aksi halde
            # yalniz "acilamadi" gorunur, kullanici parola/anahtar sorununu bilemez).
            if secrets_util.password_encrypted_but_unresolved(self.source):
                log.error("[%s] RTSP parolasi COZULEMEDI: AIEYE_SECRET_KEY (.env) "
                          "eksik veya farkli. Baska PC'ye tasirken .env dosyasini "
                          "BIRLIKTE getirin ya da kamerayi silip parolasiyla yeniden "
                          "ekleyin. Kamera baglanamayacak.", self.name)
            src = secrets_util.decrypt_url_password(self.source)
            cap = cv.VideoCapture(src, cv.CAP_FFMPEG)
            # Buffer'i kucuk tut (destekleyen backend'lerde) -> daha az gecikme
            try:
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap

    def _run(self):
        backoff = self.reconnect_min
        while not self._stop.is_set():
            self._cap = self._open()
            if not self._cap or not self._cap.isOpened():
                self._connected = False
                log.warning("[%s] acilamadi, %.1f sn sonra tekrar denenecek", self.name, backoff)
                if self._cap is not None:
                    self._cap.release()
                self._sleep(backoff)
                backoff = min(backoff * 2, self.reconnect_max)
                continue

            log.info("[%s] baglandi: %s", self.name, self._masked_source())
            self._connected = True
            backoff = self.reconnect_min
            fail_count = 0

            while not self._stop.is_set():
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    fail_count += 1
                    # Birkac ardisik bos okuma -> yayin kopmuss say, yeniden bagla
                    if fail_count >= 30:
                        log.warning("[%s] yayin koptu, yeniden baglaniliyor", self.name)
                        break
                    time.sleep(0.01)
                    continue

                fail_count = 0
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1

            self._connected = False
            if self._cap is not None:
                self._cap.release()
            if not self._stop.is_set():
                self._sleep(backoff)
                backoff = min(backoff * 2, self.reconnect_max)

        log.info("[%s] okuyucu durdu", self.name)

    def _sleep(self, seconds):
        # stop sinyalini bekleyerek uyu -> hizli kapanma
        self._stop.wait(seconds)

    def _masked_source(self):
        """Logda parolayi gizle."""
        s = str(self.source)
        if "@" in s and "://" in s:
            scheme, rest = s.split("://", 1)
            if "@" in rest:
                _creds, host = rest.split("@", 1)
                return f"{scheme}://***@{host}"
        return s


class Camera:
    """Bir mantiksal kamera: detect (sub) + opsiyonel hires (main) akislari.

    hires_url verilmezse detect akisi her ikisi icin kullanilir (tek akis modu).
    """

    def __init__(self, name, detect_source, hires_source=None, is_webcam=False):
        self.name = name
        self.single_stream = hires_source is None or hires_source == detect_source

        self.detect_reader = StreamReader(
            detect_source, name=f"{name}:detect", is_webcam=is_webcam
        )
        if self.single_stream:
            self.hires_reader = self.detect_reader
        else:
            self.hires_reader = StreamReader(
                hires_source, name=f"{name}:hires", is_webcam=is_webcam
            )

    def start(self):
        self.detect_reader.start()
        if not self.single_stream:
            self.hires_reader.start()
        return self

    def stop(self):
        self.detect_reader.stop()
        if not self.single_stream:
            self.hires_reader.stop()

    def read_detect(self):
        return self.detect_reader.read()

    def read_hires(self):
        return self.hires_reader.read()

    @property
    def connected(self):
        return self.detect_reader.connected


def build_cameras(config):
    """config sozlugunden Camera nesnelerini olustur (henuz start etme)."""
    cameras = []
    webcam_index = config.get("webcam_test_index", None)

    if webcam_index is not None:
        # Test modu: RTSP yerine yerel webcam, tek akis.
        log.info("Webcam test modu aktif (index=%s)", webcam_index)
        cameras.append(
            Camera(name="Webcam", detect_source=int(webcam_index), is_webcam=True)
        )
        return cameras

    for cam_cfg in config.get("cameras", []):
        name = cam_cfg.get("name", "Kamera")
        detect_url = cam_cfg.get("detect_url")
        hires_url = cam_cfg.get("hires_url")  # None olabilir -> tek akis
        if not detect_url:
            log.error("Kamera '%s' icin detect_url yok, atlaniyor", name)
            continue
        cameras.append(
            Camera(name=name, detect_source=detect_url, hires_source=hires_url)
        )
    return cameras
