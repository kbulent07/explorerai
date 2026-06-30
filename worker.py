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
import time

import cv2 as cv

from framing import FaceTracker, FrameTransformer
from tracking import (FaceTrackerManager, PersonTrackManager,
                      associate_faces_to_persons, compute_quality)
from detection_backend import resolve_detection_config, build_person_detector

log = logging.getLogger("facezoom.worker")


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

    def __init__(self, camera, config, db=None, on_capture=None):
        # db=None       -> yakalama DB'ye/diske yazilmaz
        # on_capture(...)-> her bitmiss gorunumde cagrilir (bellek deposu vb. icin)
        #   imza: on_capture(camera_name, crop_bgr, bbox, quality, first, last, best_t)
        self.camera = camera
        self.config = config
        self.db = db
        self.on_capture = on_capture

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

        self._frame_count = 0
        self._faces = []                 # son algilanan yuzler (detect koord.)
        self._last_detect_id = -1
        self._fps_t = time.time()
        self._fps = 0.0
        self._face_present = False

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

        self._face_present = len(self._faces) > 0

        # --- bitmiss gorunumleri isle (DB ve/veya bellek deposu) -------------
        for tr in self.manager.collect_finished(now):
            self._emit_capture(tr)

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

        # cikti boyutuna olcekle
        out_w, out_h = self.config.get("output_size", [1280, 720])
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
        self.tracker.close()
