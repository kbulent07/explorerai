# Rapor tuketici modulu: ctx['events'] icindeki izinli olaylari ReportManager'a
# iletir. Zincirde uretici modullerden (counting/recognition) SONRA yer almali.
# report_manager yoksa no-op. crop (ndarray) burada JPEG'e cevrilir; boylece
# ReportManager saf JSON-uyumlu veriyle calisir.

import cv2 as cv

from pipeline import PipelineModule
from reporting import resolve_reporting

DEFAULT_EVENTS = ("counting_crossing", "capture_finished")


class ReportingModule(PipelineModule):
    def setup(self, config, camera, services):
        self.rm = services.get("report_manager")
        cam_cfg = None
        for c in config.get("cameras") or []:
            if c.get("name") == camera:
                cam_cfg = c
                break
        rep = resolve_reporting(config, cam_cfg)
        self.event_types = set(rep.get("events") or DEFAULT_EVENTS)
        self.branch_id = rep.get("branch_id") or ""

    def process(self, ctx):
        if self.rm is None:
            return
        for ev in ctx.get("events") or []:
            if ev.get("type") not in self.event_types:
                continue
            ev = dict(ev)   # orijinali degistirme (baska tuketiciler okuyabilir)
            if self.branch_id:
                ev["branch_id"] = self.branch_id
            crop = ev.pop("crop", None)
            if crop is not None and ev.get("jpeg") is None:
                ok, buf = cv.imencode(".jpg", crop, [cv.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    ev["jpeg"] = buf.tobytes()
            self.rm.send(ev)
