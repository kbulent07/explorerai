# main.py
# -----------------------------------------------------------------------------
# FaceZoom - giriss noktasi.
#
# Coklu Hikvision RTSP kamerayi acar, dussuk cozunurluklu (sub) akista yuz algilar,
# yuksek cozunurluklu (main) akista o yuze Center Stage benzeri yumuscak DIJITAL
# zoom yapar. Her "gorunum" icin en net yuz karesini secip diske + SQLite'a yazar.
#
#   detect (sub)  -> yuz ALGILAMA (dussuk CPU)
#   hires  (main) -> canli zoom goruntusu + kaydedilecek yuksek cozunurluklu foto
#   bbox koordinatlari sub cozunurlukten hires cozunurluge oranlanir.
#
# Kisayollar:  q = cikiss,  s = anlik snapshot,  f = tam ekran ac/kapat
# -----------------------------------------------------------------------------

import logging
import os
import threading
import time

import cv2 as cv
import yaml

from camera import build_cameras
from db import Database
from worker import CameraWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("facezoom.main")


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cleanup_loop(db, retention_days, interval_hours, stop_event):
    """Periyodik retention temizligi (ayri thread)."""
    interval = max(1, interval_hours) * 3600
    while not stop_event.is_set():
        try:
            db.cleanup(retention_days)
        except Exception as e:
            log.exception("Retention temizligi hatasi: %s", e)
        stop_event.wait(interval)


def main():
    config = load_config()

    # Tek-ornek nobeti: webui.py de yakalama yapar; ayni anda ikisi calisirsa
    # kameralar cift acilir. Bloke etmez, yalniz uyarir.
    import caplock
    caplock.acquire()

    db = Database(
        db_path=config.get("db_path", "facezoom.db"),
        images_dir=config.get("images_dir", "captures"),
    )

    cameras = build_cameras(config)
    if not cameras:
        log.error("Yapilandirmada kamera yok. config.yaml'i kontrol edin.")
        return
    for cam in cameras:
        cam.start()

    workers = [CameraWorker(cam, config, db) for cam in cameras]

    show = config.get("show_windows", True)
    if show:
        for cam in cameras:
            cv.namedWindow(cam.name, cv.WINDOW_NORMAL)

    # retention temizlik thread'i
    stop_event = threading.Event()
    cleanup_thread = threading.Thread(
        target=cleanup_loop,
        args=(db, config.get("retention_days", 30),
              config.get("cleanup_interval_hours", 6), stop_event),
        daemon=True,
    )
    cleanup_thread.start()

    snapshots_dir = os.path.join(config.get("images_dir", "captures"), "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    fullscreen = False

    log.info("Baslatildi. %d kamera. Cikiss icin 'q'.", len(cameras))
    try:
        last_outputs = {}     # worker -> son uretilen kare (snapshot icin)
        while True:
            for worker in workers:
                output = worker.process()
                last_outputs[worker] = output
                if output is not None and show:
                    cv.imshow(worker.camera.name, output)

            if not show:
                # pencere yoksa CPU'yu bossa yakmamak icin minik bekleme
                time.sleep(0.005)
                continue

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                # process()'i TEKRAR cagirma; o, zoom/EMA durumunu ikinci kez
                # ilerletir ve collect_finished'i yeniden tetikleyip ayni gorunumu
                # iki kez kaydedebilirdi. Bu karede uretilmiss kareyi yaz.
                stamp = time.strftime("%Y%m%d_%H%M%S")
                for worker in workers:
                    out = last_outputs.get(worker)
                    if out is not None:
                        p = os.path.join(snapshots_dir, f"{worker.camera.name}_{stamp}.jpg")
                        cv.imwrite(p, out)
                        log.info("Snapshot: %s", p)
            elif key == ord("f"):
                fullscreen = not fullscreen
                mode = cv.WINDOW_FULLSCREEN if fullscreen else cv.WINDOW_NORMAL
                for cam in cameras:
                    cv.setWindowProperty(cam.name, cv.WND_PROP_FULLSCREEN, mode)
    except KeyboardInterrupt:
        log.info("Klavye kesintisi, kapatiliyor.")
    finally:
        stop_event.set()
        for worker in workers:
            worker.finalize()
        for cam in cameras:
            cam.stop()
        if show:
            cv.destroyAllWindows()
        db.close()
        log.info("Kapandi.")


if __name__ == "__main__":
    main()
