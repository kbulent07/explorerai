# recognition.py
# -----------------------------------------------------------------------------
# Yuz TANIMA (insightface / ArcFace) - yalniz BELLEKTE kimlik, diske YAZMAZ.
#
# Onemli: Tanima AGIRDIR ve canli akisi (zoom) BLOKE ETMEMELIDIR. Bu yuzden:
#   - Tanima yalniz "best-shot" KIRPINTISINDA (gorunum bitince) calisir, her
#     karede degil. Kirpinti kucuk oldugu icin algilama+embedding hizlidir.
#   - Calisma AYRI bir thread + kuyrukta yapilir; CameraWorker.process()
#     dongusu beklemez. Kuyruk dolarsa en eski isler dussar (canlilik onceligi).
#
# insightface modeli (buffalo_l) ilk kullanimida ~/.insightface/models altina
# bir kez iner (bu MODEL'dir; yuz/kimlik verisi degil). Embedding'ler RAM'de.
# -----------------------------------------------------------------------------

import logging
import threading
from collections import deque

import cv2 as cv

log = logging.getLogger("aieye.recognition")


class FaceRecognizer:
    """insightface FaceAnalysis sarmalayici (tembel yuklenir, thread-guvenli)."""

    def __init__(self, model_name="buffalo_l", det_size=320, min_det_score=0.5,
                 providers=None):
        self.model_name = model_name
        self.det_size = det_size
        self.min_det_score = min_det_score   # bu skorun altindaki "yuz" reddedilir
        # onnxruntime saglayicilari: GPU icin ["CUDAExecutionProvider","CPUExecutionProvider"]
        self.providers = providers or ["CPUExecutionProvider"]
        self._app = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._app is not None:
            return self._app
        with self._lock:
            if self._app is None:
                # Gec import: insightface yoksa uygulama yine de calissin
                from insightface.app import FaceAnalysis
                # CUDA saglayicisi varsa GPU cihazi (ctx_id=0), yoksa CPU (-1)
                use_gpu = any("CUDA" in p for p in self.providers)
                ctx_id = 0 if use_gpu else -1
                log.info("insightface yukleniyor (%s, %s)... ilk seferde model inebilir",
                         self.model_name, "GPU" if use_gpu else "CPU")
                app = FaceAnalysis(
                    name=self.model_name,
                    allowed_modules=["detection", "recognition"],
                    providers=self.providers,
                )
                app.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size))
                self._app = app
                log.info("insightface hazir (%s)", "GPU" if use_gpu else "CPU")
        return self._app

    def embed(self, crop_bgr):
        """BGR yuz kirpintisindan L2-normalize embedding dondurur (veya None)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        app = self._ensure()
        # Kirpinti yuze cok yakinsa (yuz kareyi kenardan kenara doldurursa)
        # insightface dedektoru kenar payi isteyip yuzu BULAMAZ -> kayit
        # sessizce dusserdi. Siyah kenarlikla pay birakarak dedektore alan ac.
        ih, iw = crop_bgr.shape[:2]
        pad = int(round(0.25 * max(ih, iw)))
        padded = cv.copyMakeBorder(
            crop_bgr, pad, pad, pad, pad,
            cv.BORDER_CONSTANT, value=(0, 0, 0),
        )
        faces = app.get(padded)
        if not faces:
            return None
        # Kirpintida birden cok yuz cikarsa en guvenli olani sec
        f = max(faces, key=lambda x: x.det_score)
        # Zayif/yanlis pozitifleri reddet -> yuz olmayan kareler elenir
        if float(f.det_score) < self.min_det_score:
            return None
        return f.normed_embedding


class RecognitionPipeline:
    """on_capture'dan gelen kirpintilari kuyruga alir; ayri thread'de embedding
    cikarip RecentFaceStore'a yazar. Canli dongu beklemez.
    """

    def __init__(self, recognizer, store, maxlen=64, require_face=True):
        self.recognizer = recognizer
        self.store = store
        # True: insightface yuz dogrulamazsa kayit SAKLANMAZ (yanlis pozitif filtresi)
        self.require_face = require_face
        self._dq = deque(maxlen=maxlen)   # dolarsa en eski dussar
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="recognition",
                                        daemon=True)
        self._dropped = 0          # require_face ile SAKLANMAYAN yakalama sayisi
        self._load_failed = False  # taniyici yuklenemedi (model/CUDA/kurulum)?

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def submit(self, camera, crop_bgr, bbox, quality, ts):
        """Canli thread'den cagrilir; HIZLIDIR (yalniz kuyruga atar)."""
        with self._cv:
            self._dq.append((camera, crop_bgr, bbox, quality, ts))
            self._cv.notify()

    def _loop(self):
        while not self._stop.is_set():
            with self._cv:
                while not self._dq and not self._stop.is_set():
                    self._cv.wait()
                if self._stop.is_set():
                    break
                camera, crop, bbox, quality, ts = self._dq.popleft()

            # --- agir kisim (kuyruk disinda) ---
            try:
                emb = self.recognizer.embed(crop)
                embed_error = False
            except Exception:
                emb = None
                embed_error = True
                if not self._load_failed:   # ilk hatanin izini BIR kez ver
                    log.exception("Yuz tanima (insightface) calistirilamadi -> "
                                  "yuzler embedding'siz saklanacak (kimlik eslesmez)")
                self._load_failed = True

            if emb is None:
                if embed_error:
                    # Taniyici CALISMADI (model inmemis / CUDA / kurulum). Yuzu
                    # DUSURME -> galeri bos kalmasin; embed'siz sakla. Bu kirpinti
                    # zaten MediaPipe yuz tespitinden gecti; require_face bir IKINCI
                    # filtredir ve o filtre bozuksa MediaPipe'a guvenmek makuldur.
                    pass
                elif self.require_face:
                    # insightface CALISTI ama kirpintida yuz DOGRULAMADI -> yanlis
                    # pozitif filtresi. Kucuk/uzak/dussuk-coz. yuzlerde SIK olur ve
                    # galeriyi bossaltir; sebebi gorulmesin diye periyodik logla.
                    self._dropped += 1
                    if self._dropped == 1 or self._dropped % 25 == 0:
                        log.warning("recognition_required=true: %d yakalama insightface "
                                    "yuz dogrulamadigi icin SAKLANMADI (galeri bos "
                                    "kalabilir). Kucuk/uzak yuzlerde beklenir; galeriyi "
                                    "doldurmak icin config.yaml'da recognition_required: "
                                    "false yapin.", self._dropped)
                    continue
            ok, buf = cv.imencode(".jpg", crop, [cv.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            self.store.add(camera, bbox, buf.tobytes(), quality, ts=ts, embedding=emb)
