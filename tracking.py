# tracking.py
# -----------------------------------------------------------------------------
# Kareler arasi yuz takibi (track_id) + "best-shot" (en net kare) skorlamasi.
#
# Amac: tek bir "gorunum" (kisi kameraya girip cikinca) icin yuzlerce kopya
# kaydetmek yerine, o gorunumun TEK ve EN NET yuz karesini secmek.
#
# YUZ TANIMA YOKTUR. Takip yalnizca kareler arasi konum/ortusme (IoU + merkez
# mesafesi) ile yapilir. track_timeout boyunca yuz gorulmezse gorunum biter ve
# en yuksek skorlu HIRES yuz kirpintisi kaydedilmek uzere dondurulur.
# -----------------------------------------------------------------------------

import logging

import cv2 as cv
import numpy as np

log = logging.getLogger("facezoom.tracking")


# ---- kalite (best-shot) skorlamasi -----------------------------------------

def sharpness_score(gray_crop):
    """Laplacian varyansi: yuksek = keskin, dussuk = bulanik. Best-shot'ta EN onemli."""
    if gray_crop is None or gray_crop.size == 0:
        return 0.0
    return float(cv.Laplacian(gray_crop, cv.CV_64F).var())


def exposure_penalty(gray_crop):
    """0..1 arasi 'iyilik' skoru. Asiri karanlik/parlak kareleri cezalandirir."""
    if gray_crop is None or gray_crop.size == 0:
        return 0.0
    mean = float(np.mean(gray_crop))
    # Ideal ~ 120 (orta ton). 0 veya 255'e yaklassinca skor dusser.
    ideal = 120.0
    score = 1.0 - min(abs(mean - ideal) / ideal, 1.0)
    return max(0.0, score)


def frontality_score(keypoints, bbox):
    """MediaPipe landmark'larindan frontallik (0..1).

    - Gozler yatayda simetrik mi (burun, iki goz ortasinda mi)
    - Burun, yuz kutusunda dikeyde makul yerde mi
    Profil/yana donuk yuzlerde dusser.
    """
    if not keypoints or "right_eye" not in keypoints or "left_eye" not in keypoints:
        return 0.5  # landmark yoksa notr
    if "nose" not in keypoints:
        return 0.5

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.5

    re = keypoints["right_eye"]
    le = keypoints["left_eye"]
    nose = keypoints["nose"]

    eyes_cx = (re[0] + le[0]) / 2.0
    # Burnun goz-merkezine yatay sapmasi, goz arasi mesafeye gore
    eye_dist = abs(le[0] - re[0]) + 1e-6
    horiz_offset = abs(nose[0] - eyes_cx) / eye_dist  # 0 = tam karssi
    horiz_score = max(0.0, 1.0 - horiz_offset)

    # Gozlerin dikey hizalanmasi (egim)
    vert_diff = abs(le[1] - re[1]) / (h + 1e-6)
    level_score = max(0.0, 1.0 - vert_diff * 3.0)

    return max(0.0, min(1.0, 0.6 * horiz_score + 0.4 * level_score))


def _normalize(value, lo, hi):
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def compute_quality(face, hires_crop, frame_area, weights):
    """Tek bir yuz icin agirlikli kalite skoru (yuksek = daha iyi).

    face       : FaceTracker.detect() ciktisindaki sozluk (bbox/conf/keypoints)
                 NOT: bbox burada HIRES koordinatlarinda beklenir.
    hires_crop : ayni yuzun hires akistan kirpilmiss BGR goruntusu
    frame_area : hires kare alani (yuz boyut oranini normalize etmek icin)
    weights    : config'ten gelen agirlik sozlugu
    """
    if hires_crop is None or hires_crop.size == 0:
        return 0.0

    gray = cv.cvtColor(hires_crop, cv.COLOR_BGR2GRAY)

    # 1) Netlik (Laplacian varyansi) - genelde 0..~1000+; 300'de doygunluk varsay
    sharp = _normalize(sharpness_score(gray), 0.0, 300.0)

    # 2) Yuz boyutu (kareye orani) - buyuk yuz = daha yuksek cozunurluk
    x, y, w, h = face["bbox"]
    size_ratio = (w * h) / (frame_area + 1e-6)
    size = _normalize(size_ratio, 0.0, 0.25)  # karenin ~%25'i ve ustu = tam puan

    # 3) Frontallik
    frontal = frontality_score(face.get("keypoints"), face["bbox"])

    # 4) Pozlama
    exposure = exposure_penalty(gray)

    # 5) Algilama guveni
    conf = float(face.get("confidence", 0.0))

    score = (
        weights.get("sharpness", 0.45) * sharp
        + weights.get("size", 0.20) * size
        + weights.get("frontality", 0.15) * frontal
        + weights.get("exposure", 0.10) * exposure
        + weights.get("confidence", 0.10) * conf
    )
    return float(score)


# ---- takip (track_id atama) -------------------------------------------------

def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center(bbox):
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def associate_faces_to_persons(faces, person_tracks):
    """Her yuzu, merkezini iceren kisi kutusuna esle. Coklu adayda en yuksek
    IoU'lu kisi (esitlikte en kucuk track_id). Iceren kisisi olmayan yuz duser.

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


class Track:
    """Tek bir gorunum: bir yuzun kameradaki yassam suresi + en iyi karesi."""

    __slots__ = (
        "track_id", "bbox", "first_seen", "last_seen",
        "best_score", "best_crop", "best_time", "best_bbox",
    )

    def __init__(self, track_id, bbox, now):
        self.track_id = track_id
        self.bbox = bbox            # son gorulen detect bbox (sub koordinati)
        self.first_seen = now
        self.last_seen = now
        self.best_score = -1.0
        self.best_crop = None       # en iyi HIRES kirpinti (BGR)
        self.best_time = now
        self.best_bbox = bbox

    def maybe_update_best(self, score, crop, bbox, now):
        if score > self.best_score:
            self.best_score = score
            self.best_crop = crop
            self.best_bbox = bbox
            self.best_time = now


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
        """track_timeout'u asan track'leri dondur ve listeden cikar.

        Donus: bitmiss Track listesi (en az bir best_crop'u olanlar finalize
        edilmek uzere; crop'u olmayanlar sessizce atilir).
        """
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
        import warnings
        # TODO: supervision ileride ByteTrack'i yeniden adlandirabilir (pydeprecate
        #       FutureWarning). Surum yukseltince bu bastirma kaldirilip yeni API'ye
        #       gecilmeli.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
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


class FaceTrackerManager(BestShotTrackManager):
    """Detect bbox'lari kareler arasi eslestirip track_id atar; best-shot tutar.

    track_timeout boyunca guncellenmeyen track'ler "biter" ve finalize edilir.
    """

    def __init__(self, iou_threshold=0.3, max_center_dist=120.0, track_timeout=2.0):
        super().__init__(track_timeout=track_timeout)
        self.iou_threshold = iou_threshold
        self.max_center_dist = max_center_dist

    def update(self, detect_bboxes, now):
        """Mevcut karedeki detect bbox'larini track'lere esle. Eslesmeyenlere
        yeni track ac. Donus: [(track_id, bbox), ...] (bu karedeki eslessmeler).

        Eslessme BELIRLEYICIDIR: aday track'ler arasinda en yuksek birlessik skor
        (IoU agirlikli + merkez yakinligi) secilir; esitlikte en kucuk track_id.
        Onceki surumde yalniz mesafeyle eslesen adaylar 'metric=0 >= 0' nedeniyle
        SET gezinme sirasina gore (belirsiz) seciliyordu -> yakin yuzlerde ID
        atlamalari olabiliyordu.
        """
        assignments = []
        unmatched = set(self.tracks.keys())

        for bbox in detect_bboxes:
            best_tid = None
            best_score = 0.0
            cx, cy = _center(bbox)

            # Belirlilik icin track_id'ye gore sirali gez (esitlik -> en kucuk id)
            for tid in sorted(unmatched):
                tr = self.tracks[tid]
                iou = _iou(bbox, tr.bbox)
                tcx, tcy = _center(tr.bbox)
                dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
                # Aday olma kapisi: IoU yeterli VEYA merkez yeterince yakin
                if iou < self.iou_threshold and dist > self.max_center_dist:
                    continue
                # Birlessik skor: IoU baskindir, merkez yakinligi ek katki verir.
                # Boylece mesafeyle eslesen adaylar arasinda EN YAKINI secilir.
                closeness = max(0.0, 1.0 - dist / max(self.max_center_dist, 1e-6))
                score = iou + 0.5 * closeness
                if score > best_score:
                    best_score = score
                    best_tid = tid

            if best_tid is not None:
                tr = self.tracks[best_tid]
                tr.bbox = bbox
                tr.last_seen = now
                unmatched.discard(best_tid)
                assignments.append((best_tid, bbox))
            else:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(tid, bbox, now)
                assignments.append((tid, bbox))

        return assignments
