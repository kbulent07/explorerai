# tests/test_module_zoom.py
import numpy as np

from modules.zoom import ZoomModule


def test_zoom_kapaliyken_output_degismez():
    m = ZoomModule()
    m.setup({"zoom_enabled": False, "output_size": [200, 100]}, "Kam", {})
    frame = np.full((100, 200, 3), 7, dtype=np.uint8)
    ctx = {"output": frame, "hires_dims": (200, 100), "faces": [],
           "scale": (1.0, 1.0), "now": 1.0}
    m.draw(ctx)
    assert ctx.get("zoomed") in (False, None)
    assert np.array_equal(ctx["output"], frame)   # dokunmadi


def test_zoom_acikken_zoomed_bayragini_yazar():
    m = ZoomModule()
    m.setup({"zoom_enabled": True, "zoom_factor": 2.0}, "Kam", {})
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    ctx = {"output": frame, "hires_dims": (200, 100),
           "faces": [{"bbox": (80, 40, 40, 20)}], "scale": (1.0, 1.0), "now": 1.0}
    m.draw(ctx)
    assert "zoomed" in ctx            # bayrak set edildi (True/False FrameTransformer'a bagli)
    assert ctx["output"].shape[0] > 0


def test_set_enabled_calisir():
    m = ZoomModule()
    m.setup({"zoom_enabled": True}, "Kam", {})
    m.set_enabled(False)
    assert m.enabled is False
