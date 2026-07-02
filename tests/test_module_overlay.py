# tests/test_module_overlay.py
import numpy as np

from modules.overlay import OverlayModule


def test_overlay_output_uzerine_cizer():
    m = OverlayModule()
    m.setup({"debug_overlay": True}, "Kam", {})
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    ctx = {"output": frame, "camera": "Kam", "fps": 12.3,
           "face_present": True, "zoomed": False}
    before = frame.copy()
    m.draw(ctx)
    # ust seride (overlay bandi) pikseller degismis olmali
    assert not np.array_equal(before[:28], ctx["output"][:28])


def test_overlay_kapaliyken_cizmez():
    m = OverlayModule()
    m.setup({"debug_overlay": False}, "Kam", {})
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    before = frame.copy()
    m.draw({"output": frame, "camera": "Kam", "fps": 1.0})
    assert np.array_equal(before, frame)
