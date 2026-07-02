# modules/counting.py
# Giris/cikis sayim modulu: ctx['tracks']'te cizgi-gecis tespiti -> CountingStore'a
# isimsiz olay + (varsa) asenkron isim cozumu. worker._run_counting'in modul hali.

import logging

import cv2 as cv

from pipeline import PipelineModule
from worker import scale_bbox, crop_with_margin

log = logging.getLogger("aieye.modules.counting")


class CountingModule(PipelineModule):
    def setup(self, config, camera, services):
        self.line_counter = services.get("line_counter")
        self.counting_store = services.get("counting_store")
        self.name_resolver = services.get("name_resolver")
        self.manager = services.get("manager")

    def process(self, ctx):
        if self.line_counter is None or self.counting_store is None:
            return
        tracks = ctx.get("tracks") or []
        dims = ctx.get("detect_dims")
        self.counting_store.note(len(tracks))
        events = self.line_counter.update(tracks, dims)
        if not events:
            return
        bbox_by_tid = dict(tracks)
        tid_face = ctx.get("tid_face") or {}
        sx, sy = ctx.get("scale", (1.0, 1.0))
        hires = ctx.get("hires_frame")
        now = ctx.get("now")
        for tid, direction in events:
            jpeg = None
            tr = self.manager.tracks.get(tid) if self.manager is not None else None
            fcrop = getattr(tr, "best_crop", None) if tr is not None else None
            if fcrop is None:
                face = tid_face.get(tid)
                if face is not None:
                    fcrop = crop_with_margin(hires, scale_bbox(face["bbox"], sx, sy), 0.3)
            crop = fcrop
            if crop is None:
                dbbox = bbox_by_tid.get(tid)
                if dbbox is not None:
                    crop = crop_with_margin(hires, scale_bbox(dbbox, sx, sy), 0.1)
            if crop is not None:
                ok, buf = cv.imencode(".jpg", crop, [cv.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    jpeg = buf.tobytes()
            # Rapor katmanina blackboard olayi (ReportingModule tuketir).
            # Isim asenkron cozulur -> ilk raporda None gider (YAGNI, spec §2).
            ctx.setdefault("events", []).append({
                "type": "counting_crossing", "camera": ctx.get("camera"),
                "ts": now, "direction": "in" if direction == "giris" else "out",
                "name": None, "jpeg": jpeg,
            })
            eid = self.counting_store.record(direction, ts=now, name=None,
                                             camera=ctx.get("camera"), jpeg=jpeg)
            if eid is not None and fcrop is not None and self.name_resolver is not None:
                self.name_resolver.submit(eid, fcrop, ctx.get("camera"), now)
