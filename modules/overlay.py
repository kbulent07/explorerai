# modules/overlay.py
# Debug overlay modulu: FPS + yuz durumu + kamera adi + ZOOM etiketi. draw() ile
# ctx['output'] uzerine cizer. (worker._draw_overlay'in modul karsiligi.)

import cv2 as cv

from pipeline import PipelineModule


class OverlayModule(PipelineModule):
    def setup(self, config, camera, services):
        self.enabled = bool(config.get("debug_overlay", True))

    def draw(self, ctx):
        if not getattr(self, "enabled", True):
            return
        frame = ctx.get("output")
        if frame is None:
            return
        face_present = ctx.get("face_present", False)
        status = "YUZ VAR" if face_present else "yuz yok"
        color = (0, 220, 0) if face_present else (0, 165, 255)
        conn = "" if ctx.get("connected", True) else "  [BAGLANTI YOK]"
        text = f"{ctx.get('camera','')}  FPS:{ctx.get('fps',0.0):4.1f}  {status}{conn}"
        cv.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
        cv.putText(frame, text, (8, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if ctx.get("zoomed"):
            cv.putText(frame, "ZOOM", (frame.shape[1] - 90, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
