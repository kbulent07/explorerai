# modules/zoom.py
# Canli dijital pan-zoom modulu: ctx['output']'u en buyuk yuze odakli olcekler.
# FrameTransformer durumu (EMA/hold) modul icinde tutulur. (worker'daki zoom
# blogunun modul karsiligi.) set_enabled ile canli ac/kapat (LiveManager.set_zoom).

from pipeline import PipelineModule
from framing import FrameTransformer
from worker import scale_bbox


class ZoomModule(PipelineModule):
    def setup(self, config, camera, services):
        self.enabled = bool(config.get("zoom_enabled", True))
        self._zoom_factor = config.get("zoom_factor", 2.5)
        self._smoothing = config.get("smoothing", 0.15)
        self._hold = config.get("hold_seconds", 1.5)
        self._tf = None

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)

    def _ensure_tf(self, fw, fh):
        if self._tf is None:
            self._tf = FrameTransformer(fw, fh, zoom_factor=self._zoom_factor,
                                        smoothing=self._smoothing,
                                        hold_seconds=self._hold)
        elif (self._tf.frame_width, self._tf.frame_height) != (fw, fh):
            self._tf.update_size(fw, fh)

    def draw(self, ctx):
        if not self.enabled:
            ctx["zoomed"] = False
            return
        frame = ctx.get("output")
        if frame is None:
            return
        hw, hh = ctx.get("hires_dims", (frame.shape[1], frame.shape[0]))
        self._ensure_tf(hw, hh)
        target = None
        faces = ctx.get("faces") or []
        if faces:
            sx, sy = ctx.get("scale", (1.0, 1.0))
            largest = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
            target = scale_bbox(largest["bbox"], sx, sy)
        out, zoomed = self._tf.transform(frame, target, now=ctx.get("now"))
        ctx["output"] = out
        ctx["zoomed"] = zoomed
