# worker.py
# -----------------------------------------------------------------------------
# Yakalama hatti cekirdegi: CameraWorker.
#
# Tek bir kamera icin algilama/zoom/takip/best-shot durumunu tutar. Hem CLI/GUI
# girisi (main.py) hem web girisi (webui.py -> live.py) bu modulu kullanir; bu
# sayede iki giris noktasi birbirine BAGIMLI degildir (onceden live.py main.py'yi
# import ediyordu -> ters katmanlama).
#
#   detect (sub)  -> yuz ALGILAMA (dussuk CPU)
#   hires  (main) -> canli zoom goruntusu + kaydedilecek yuksek cozunurluklu foto
#   bbox koordinatlari sub cozunurlukten hires cozunurluge oranlanir.
# -----------------------------------------------------------------------------

import logging
import threading
import time
from collections import deque

import cv2 as cv

from framing import FaceTracker, FrameTransformer
from tracking import (FaceTrackerManager, PersonTrackManager,
                      associate_faces_to_persons, compute_quality)
from detection_backend import resolve_detection_config, build_person_detector

log = logging.getLogger("facezoom.worker")


class _NameResolver:
    """Gecis aninda yuz -> isim cozumunu ANA DONGUDEN ayirir.

    name_provider bir ArcFace embed'i calistirir (AGIR). Ana process() dongusunde
    senkron cagrilirsa, ard arda gelen gecislerde canli akista kare takilmasina
    yol acar. Bu yuzden cozum ayri thread + kucuk kuyrukta yapilir; kuyruk dolarsa
    en eski is dusser (canlilik onceligi). Cozulen isim CountingStore olayina
    (eid) yazilir (set_name)."""

    def __init__(self, name_provider, counting_store, maxlen=32):
        self._name_provider = name_provider
        self._store = counting_store
        self._dq = deque(maxlen=maxlen)   # dolarsa en eski dusser
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="name-resolver",
                                        daemon=True)
        self._thread.start()

    def submit(self, eid, crop_bgr, camera, ts):
        """Ana thread'den cagrilir; HIZLIDIR (yalniz kuyruga atar)."""
        with self._cv:
            self._dq.append((eid, crop_bgr, camera, ts))
            self._cv.notify()

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def _loop(self):
        while not self._stop.is_set():
            with self._cv:
                while not self._dq and not self._stop.is_set():
                    self._cv.wait()
                if self._stop.is_set():
                    break
                eid, crop, camera, ts = self._dq.popleft()
            try:
                name = self._name_provider(crop, camera, ts)
            except Exception:
                log.exception("name_provider hatasi (async)")
                name = None
            if name:
                self._store.set_name(eid, name)


def scale_bbox(bbox, sx, sy):
    """detect (sub) koordinatlarindaki bbox'u hires koordinatlarina oranla."""
    x, y, w, h = bbox
    return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))


def crop_with_margin(frame, bbox, margin=0.3):
    """bbox'i kenar payiyla genisletip kareden kirp (yuz + biraz cevre)."""
    fh, fw = frame.shape[:2]
    x, y, w, h = bbox
    mx, my = int(w * margin), int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(fw, x + w + mx)
    y2 = min(fh, y + h + my)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


class CameraWorker:
    """Tek bir kamera icin algilama/zoom/takip durumunu tutar.

    Tum kameralar ana dongude SIRAYLA islenir; RTSP okuma zaten ayri thread'de
    oldugundan bu donguyu bloke etmez.
    """

    def __init__(self, camera, config, db=None, on_capture=None,
                 line_counter=None, counting_store=None, name_provider=None):
        # db=None       -> yakalama DB'ye/diske yazilmaz
        # on_capture(...)-> her bitmiss gorunumde cagrilir (bellek deposu vb. icin)
        #   imza: on_capture(camera_name, crop_bgr, bbox, quality, first, last, best_t)
        # line_counter/counting_store -> yalniz GIRIS/CIKIS sayim kamerasinda verilir;
        #   None ise sayim YOK (diger kameralar etkilenmez).
        self.camera = camera
        self.config = config
        self.db = db
        self.on_capture = on_capture
        self.line_counter = line_counter
        self.counting_store = counting_store
        # name_provider(face_crop_bgr) -> isim | None. Gecis aninda track'in yuzunu
        # kimlige baglamak icin (webui: ArcFace + RECENT). Worker tanimaya bagli degil.
        self.name_provider = name_provider

        self.tracker = FaceTracker(
            min_detection_confidence=config.get("detection_confidence", 0.6),
            model_path=config.get("face_model"),  # None -> framing varsayilani
        )
        # Tepeden/uzak kameralarda yuzler sub-stream'de cok kucuk kalir; bu secenek
        # algilamayi yuksek cozunurluklu (hires) akista yapar (biraz daha CPU).
        self.detect_on_hires = bool(config.get("detect_on_hires", False))
        # Algilamayi kucultulmuss karede yap (CPU): 0.5 = yari boyut. 1.0 = kapali.
        self.detect_downscale = float(config.get("detect_downscale", 0.5))
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
                frame_rate=max(1, int(config.get("preview_fps", 12) / max(1, int(config.get("detect_interval", 2))))),
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
        # Merkez-mesafe eslessme esigi SABIT 120px degil; algilama karesinin
        # GENISLIGINE oranli (cozunurluk degisince tutarli kalir). process()
        # icinde dw belli olunca her karede guncellenir.
        self._center_dist_factor = float(config.get("track_center_dist_factor", 0.2))
        # FrameTransformer ilk hires kare gelince (boyut belli olunca) kurulur
        self.transformer = None

        self.weights = config.get("quality_weights", {})
        self.min_face_size = config.get("min_face_size", 40)
        # Kaydedilen yuz karesinde yuz cevresinde birakilan pay (yuksek = daha
        # genis kadraj, daha az yuze-yakin kirpma). bbox genissliginin orani.
        self.crop_margin = config.get("crop_margin", 0.6)
        self.detect_interval = max(1, config.get("detect_interval", 2))
        # Canli goruntuye dijital pan-zoom uygula (Center Stage). Kapaliysa tam
        # kare gosterilir (sag/sol zoom yok). Yakalama/best-shot bundan ETKILENMEZ.
        self.zoom_enabled = bool(config.get("zoom_enabled", True))
        # Sayim kamerasinda zoom KAPALI tutulur: canli cikti tam kare olur, boylece
        # normalize cizgi (UI'da cizilen) ile detect uzayi ayni cercevelemede kalir.
        if self.line_counter is not None:
            self.zoom_enabled = False

        self._frame_count = 0
        self._faces = []                 # son algilanan yuzler (detect koord.)
        self._last_detect_id = -1
        self._fps_t = time.time()
        self._fps = 0.0
        self._face_present = False

        # Gecis isim cozumunu ana donguden ayir (yalniz sayim + isim saglayici varsa).
        self._name_resolver = None
        if self.name_provider is not None and self.counting_store is not None:
            self._name_resolver = _NameResolver(self.name_provider, self.counting_store)

        # Yakalama hatti teshisi: yuzler NEREDE kayboluyor gorunur olsun (galeri
        # bos kalinca sebep bulmak icin). Periyodik (30 sn) INFO loglanir.
        self._diag = {"cycles": 0, "faces": 0, "paired": 0, "emitted": 0}
        self._diag_t = time.time()

    def _ensure_transformer(self, fw, fh):
        if self.transformer is None:
            self.transformer = FrameTransformer(
                fw, fh,
                zoom_factor=self.config.get("zoom_factor", 2.5),
                smoothing=self.config.get("smoothing", 0.15),
                hold_seconds=self.config.get("hold_seconds", 1.5),
            )
        elif (self.transformer.frame_width, self.transformer.frame_height) != (fw, fh):
            self.transformer.update_size(fw, fh)

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

    def process(self):
        """Bir adim isle, gosterilecek (output_frame) dondur veya None."""
        now = time.time()
        detect_frame, detect_id = self.camera.read_detect()
        hires_frame, hires_id = self.camera.read_hires()

        if detect_frame is None and hires_frame is None:
            return None
        # hires yoksa (tek akis ya da henuz gelmedi) detect'i hires gibi kullan
        if hires_frame is None:
            hires_frame, hires_id = detect_frame, detect_id
        if detect_frame is None:
            detect_frame, detect_id = hires_frame, hires_id

        hh, hw = hires_frame.shape[:2]

        # Algilama koordinat uzayi + giris goruntusu:
        #  - detect_on_hires: hires'i detect_downscale ile kucult (CPU tasarrufu);
        #    bbox'lar bu kucuk uzayda, sx/sy ile hires'e olceklenir.
        #  - aksi halde: sub-stream'i oldugu gibi kullan.
        if self.detect_on_hires:
            ds = self.detect_downscale if 0.1 < self.detect_downscale < 1.0 else 1.0
            dw, dh = max(1, int(hw * ds)), max(1, int(hh * ds))
            det_source, det_id, det_resize = hires_frame, hires_id, (ds < 1.0)
        else:
            dh, dw = detect_frame.shape[:2]
            det_source, det_id, det_resize = detect_frame, detect_id, False
        sx, sy = hw / dw, hh / dh

        # Merkez-mesafe esigi yalniz mediapipe (FaceTrackerManager) yolunda anlamli
        if self.backend == "mediapipe":
            self.manager.max_center_dist = max(1.0, self._center_dist_factor * dw)

        self._ensure_transformer(hw, hh)
        self._frame_count += 1

        # --- algilama (her detect_interval karede bir, yeni kare geldiyse) ----
        run_detect = (self._frame_count % self.detect_interval == 0) and (det_id != self._last_detect_id)
        if run_detect:
            self._last_detect_id = det_id
            det_img = cv.resize(det_source, (dw, dh)) if det_resize else det_source

            # Yuz tespiti her iki arka ucta da gerekli (best-shot + frontallik)
            faces = self.tracker.detect(det_img)
            faces = [f for f in faces if f["bbox"][3] >= self.min_face_size]
            self._faces = faces
            frame_area = float(hw * hh)
            self._diag["cycles"] += 1
            self._diag["faces"] += len(faces)

            tracks = []       # [(track_id, (x,y,w,h))] detect koord. (sayim icin)
            tid_face = {}     # track_id -> face dict (sayim isim eslemesi icin)
            if self.backend == "yolox_person":
                # 1) YOLOX kisi tespiti -> 2) ByteTrack track_id
                persons = self._person_detector.detect(det_img)
                person_tracks = self.manager.update(persons, now)
                tracks = person_tracks
                # 3) yuzleri iceren kisiye esle -> kisi track_id devralir
                pairs = associate_faces_to_persons(faces, person_tracks)
                dropped = len(faces) - len(pairs)
                if dropped > 0:
                    log.debug("%d yuz iceren kisi kutusu bulunamadi (dusuruldu)", dropped)
                self._diag["paired"] += len(pairs)
                for tid, face in pairs:
                    tid_face[tid] = face
                    self._record_best(tid, face, sx, sy, hires_frame, frame_area, now)
            else:
                # mediapipe: yuz bbox'lari dogrudan track edilir (mevcut davranis)
                detect_bboxes = [f["bbox"] for f in faces]
                assignments = self.manager.update(detect_bboxes, now)
                tracks = assignments
                bbox_to_face = {tuple(f["bbox"]): f for f in faces}
                for tid, dbbox in assignments:
                    face = bbox_to_face.get(tuple(dbbox))
                    if face is None:
                        continue
                    tid_face[tid] = face
                    self._record_best(tid, face, sx, sy, hires_frame, frame_area, now)

            # --- GIRIS/CIKIS sayimi (yalniz sayim kamerasinda; backend-bagimsiz) --
            self._run_counting(tracks, (dw, dh), tid_face, sx, sy, hires_frame, now)

        self._face_present = len(self._faces) > 0

        # --- bitmiss gorunumleri isle (DB ve/veya bellek deposu) -------------
        finished = self.manager.collect_finished(now)
        for tr in finished:
            self._emit_capture(tr)
        self._diag["emitted"] += len(finished)

        # --- yakalama teshisi (periyodik) -> yuzler nerede kayboluyor? -------
        if now - self._diag_t >= 30.0:
            d = self._diag
            hint = ""
            if d["cycles"] > 0 and d["faces"] == 0:
                hint = " -> MediaPipe YUZ BULAMIYOR (tepeden/uzak/dussuk-coz.; " \
                       "detect_on_hires: true veya cpu_profile: normal/high, min_face_size dussur)"
            elif self.backend == "yolox_person" and d["faces"] > 0 and d["paired"] == 0:
                hint = " -> yuzler kisi kutusuna ESLESMIYOR"
            elif d["faces"] > 0 and d["emitted"] == 0:
                hint = " -> best-shot emit edilmedi (track bitmiyor?)"
            log.info("[%s] yakalama tani(30s): %d dongu, %d yuz, %d eslesme, %d best-shot%s",
                     self.camera.name, d["cycles"], d["faces"], d["paired"], d["emitted"], hint)
            self._diag = {"cycles": 0, "faces": 0, "paired": 0, "emitted": 0}
            self._diag_t = now

        # --- canli zoom (en buyuk = en yakin yuze odaklan) -------------------
        # zoom_enabled False -> tam kareyi oldugu gibi goster (pan/zoom yok).
        if self.zoom_enabled:
            target_hbbox = None
            if self._faces:
                largest = max(self._faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
                target_hbbox = scale_bbox(largest["bbox"], sx, sy)
            output, zoomed = self.transformer.transform(hires_frame, target_hbbox, now=now)
        else:
            output, zoomed = hires_frame, False

        # cikti boyutuna olcekle (biçimsiz output_size -> guvenli varsayilan)
        osz = self.config.get("output_size", [1280, 720])
        if not (isinstance(osz, (list, tuple)) and len(osz) == 2):
            osz = [1280, 720]
        out_w, out_h = int(osz[0]), int(osz[1])
        if (output.shape[1], output.shape[0]) != (out_w, out_h):
            output = cv.resize(output, (out_w, out_h), interpolation=cv.INTER_LINEAR)

        # --- FPS ---
        dt = now - self._fps_t
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._fps_t = now

        if self.config.get("debug_overlay", True):
            self._draw_overlay(output, zoomed)

        return output

    def _run_counting(self, tracks, dims, tid_face, sx, sy, hires_frame, now):
        """Sayim kamerasinda track merkezlerini cizgi-gecis sayacina ver; gecen
        kisi icin kucuk resim kirp, mumkunse yuzu kimlige (isim) bagla, depoya yaz.
        Sayac yoksa hizlica doner. dims=(dw,dh) detect uzayi."""
        if self.line_counter is None or self.counting_store is None:
            return
        self.counting_store.note(len(tracks))   # teshis: kac track goruldu
        events = self.line_counter.update(tracks, dims)
        if not events:
            return
        bbox_by_tid = dict(tracks)
        for tid, direction in events:
            jpeg = None
            name = None
            # ISIM icin: track boyunca yakalanan EN IYI yuz kirpintisini kullan
            # (o karedeki yuz degil). Yan-yana/acisi-kacan kisi bir an bile yuzunu
            # gostermisse best_crop onu tutar -> isim eslemesi cok daha saglam.
            tr = self.manager.tracks.get(tid)
            fcrop = getattr(tr, "best_crop", None) if tr is not None else None
            if fcrop is None:
                face = tid_face.get(tid)   # en iyi yuz yoksa bu karedeki yuze dus
                if face is not None:
                    fcrop = crop_with_margin(hires_frame, scale_bbox(face["bbox"], sx, sy), 0.3)
            crop = fcrop
            if crop is None:
                # yuz hic yoksa govde kutusundan kucuk resim
                dbbox = bbox_by_tid.get(tid)
                if dbbox is not None:
                    crop = crop_with_margin(hires_frame, scale_bbox(dbbox, sx, sy), 0.1)
            if crop is not None:
                ok, buf = cv.imencode(".jpg", crop, [cv.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    jpeg = buf.tobytes()
            # Olayi ONCE isimsiz kaydet (canliligi bloke etme). Isim cozumu
            # (ArcFace embed) AGIR -> ayri thread'de coz, cozulunce olayin adini
            # set_name ile guncelle. (name burada None; asenkron doldurulur.)
            eid = self.counting_store.record(direction, ts=now, name=name,
                                             camera=self.camera.name, jpeg=jpeg)
            if eid is not None and fcrop is not None and self._name_resolver is not None:
                self._name_resolver.submit(eid, fcrop, self.camera.name, now)

    def _emit_capture(self, tr):
        """Bitmiss bir gorunumun en-net karesini DB'ye ve/veya callback'e ilet."""
        if self.db is not None:
            self.db.save_capture(
                camera_name=self.camera.name,
                crop_bgr=tr.best_crop,
                quality_score=tr.best_score,
                first_seen=tr.first_seen,
                last_seen=tr.last_seen,
                best_time=tr.best_time,
            )
        if self.on_capture is not None:
            try:
                self.on_capture(self.camera.name, tr.best_crop, tr.best_bbox,
                                tr.best_score, tr.first_seen, tr.last_seen, tr.best_time)
            except Exception:
                log.exception("on_capture callback hatasi")

    def _draw_overlay(self, frame, zoomed):
        status = "YUZ VAR" if self._face_present else "yuz yok"
        color = (0, 220, 0) if self._face_present else (0, 165, 255)
        conn = "" if self.camera.connected else "  [BAGLANTI YOK]"
        text = f"{self.camera.name}  FPS:{self._fps:4.1f}  {status}{conn}"
        cv.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
        cv.putText(frame, text, (8, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if zoomed:
            cv.putText(frame, "ZOOM", (frame.shape[1] - 90, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

    def set_zoom(self, enabled):
        """Canli dijital pan-zoom'u ac/kapat (calissirken degissir)."""
        self.zoom_enabled = bool(enabled)

    def finalize(self):
        """Kapanissta aktif gorunumleri kaydet ve kaynaklari birak."""
        now = time.time()
        for tr in self.manager.flush_all(now):
            self._emit_capture(tr)
        if self._name_resolver is not None:
            self._name_resolver.stop()
        self.tracker.close()
