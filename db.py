# db.py
# -----------------------------------------------------------------------------
# SQLite veritabani: yakalanan en-net yuz kareleri (captures).
#
# Yuz GORUNTUSU diske dosya olarak yazilir; veritabaninda yalniz YOLU tutulur.
# Bu sayede DB kucuk/hizli kalir, dosyalar dogrudan galeriden servis edilir.
#
# retention_days'ten eski kayitlar ve onlarin dosyalari periyodik olarak silinir
# (KVKK / veri saklama suresi).
#
# SQLite sunucusuzdur; ayri bir veritabani sunucusu gerektirmez.
# -----------------------------------------------------------------------------

import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta

import cv2 as cv

log = logging.getLogger("aieye.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS captures (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_name       TEXT    NOT NULL,
    first_seen        TEXT    NOT NULL,   -- ISO 8601 timestamp
    last_seen         TEXT    NOT NULL,
    best_capture_time TEXT    NOT NULL,
    image_path        TEXT    NOT NULL,
    quality_score     REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_captures_time ON captures(best_capture_time);
CREATE INDEX IF NOT EXISTS idx_captures_camera ON captures(camera_name);
"""


def ts_to_iso(ts):
    """Unix timestamp (float) -> ISO 8601 string."""
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


class Database:
    """captures tablosu icin ince bir sarmalayici (thread-guvenli)."""

    def __init__(self, db_path="aieye.db", images_dir="captures"):
        self.db_path = db_path
        self.images_dir = images_dir
        os.makedirs(self.images_dir, exist_ok=True)

        self._lock = threading.Lock()
        # check_same_thread=False: yazma main thread'den, okuma webui'den olabilir.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ---- yazma --------------------------------------------------------------

    def save_capture(self, camera_name, crop_bgr, quality_score,
                     first_seen, last_seen, best_time):
        """En-net yuz kirpintisini diske yaz, kaydi DB'ye ekle. id dondurur."""
        dt = datetime.fromtimestamp(best_time)
        safe_cam = "".join(c if c.isalnum() else "_" for c in camera_name)
        fname = f"{safe_cam}_{dt.strftime('%Y%m%d_%H%M%S')}_{int(best_time*1000)%1000:03d}.jpg"
        # Tarihe gore alt klasor -> klasor basina dosya sayisi kontrollu
        subdir = os.path.join(self.images_dir, dt.strftime("%Y-%m-%d"))
        os.makedirs(subdir, exist_ok=True)
        abs_path = os.path.join(subdir, fname)

        ok = cv.imwrite(abs_path, crop_bgr)
        if not ok:
            log.error("Yuz goruntusu yazilamadi: %s", abs_path)
            return None

        # DB'de proje koküne gore goreli yol tut -> tassinabilir. Farkli surucude
        # (Windows) goreli yol uretilemezse mutlak yola dus.
        try:
            rel_path = os.path.relpath(abs_path, start=".").replace("\\", "/")
        except ValueError:
            rel_path = os.path.abspath(abs_path).replace("\\", "/")

        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO captures
                   (camera_name, first_seen, last_seen, best_capture_time,
                    image_path, quality_score)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    camera_name,
                    ts_to_iso(first_seen),
                    ts_to_iso(last_seen),
                    ts_to_iso(best_time),
                    rel_path,
                    float(quality_score),
                ),
            )
            self._conn.commit()
            cap_id = cur.lastrowid

        log.info("Kayit eklendi #%d [%s] skor=%.3f -> %s",
                 cap_id, camera_name, quality_score, rel_path)
        return cap_id

    # NOT: Galeri (DB okuma) arayuzu bellek-ici moda gecisle kaldirildi; eski
    # query_captures/get_capture/list_cameras okuma metotlari hicbir yerden
    # cagrilmadigi icin (olu kod) temizlendi. main.py yalniz save_capture +
    # cleanup kullanir. Gerekirse git gecmisinden geri alinabilir.

    # ---- retention (KVKK temizligi) ----------------------------------------

    def cleanup(self, retention_days):
        """retention_days'ten eski kayitlari VE dosyalarini sil. Silinen sayi."""
        if not retention_days or retention_days <= 0:
            return 0
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat(timespec="seconds")

        with self._lock:
            rows = self._conn.execute(
                "SELECT id, image_path FROM captures WHERE best_capture_time < ?",
                (cutoff,),
            ).fetchall()

            for r in rows:
                path = r["image_path"]
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except OSError as e:
                    log.warning("Dosya silinemedi %s: %s", path, e)

            if rows:
                ids = [r["id"] for r in rows]
                self._conn.executemany(
                    "DELETE FROM captures WHERE id = ?", [(i,) for i in ids]
                )
                self._conn.commit()

        if rows:
            log.info("Retention: %d eski kayit ve dosyasi silindi (> %d gun)",
                     len(rows), retention_days)
        return len(rows)

    def close(self):
        with self._lock:
            self._conn.close()
