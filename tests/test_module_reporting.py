import numpy as np

from modules.reporting import ReportingModule


class _FakeRM:
    def __init__(self):
        self.sent = []
    def send(self, ev):
        self.sent.append(ev)
        return True


def test_filtre_yalniz_izinli_tipler():
    rm = _FakeRM()
    m = ReportingModule()
    m.setup({"reporting": {"events": ["counting_crossing"]}}, "Kam",
            {"report_manager": rm})
    m.process({"events": [
        {"type": "counting_crossing", "camera": "Kam", "ts": 1.0},
        {"type": "capture_finished", "camera": "Kam", "ts": 1.0},
    ]})
    assert len(rm.sent) == 1
    assert rm.sent[0]["type"] == "counting_crossing"


def test_rm_yoksa_noop():
    m = ReportingModule()
    m.setup({}, "Kam", {"report_manager": None})
    m.process({"events": [{"type": "counting_crossing"}]})   # patlamamali


def test_capture_crop_jpege_cevrilir():
    rm = _FakeRM()
    m = ReportingModule()
    m.setup({}, "Kam", {"report_manager": rm})
    crop = np.zeros((8, 8, 3), np.uint8)
    ev_in = {"type": "capture_finished", "camera": "Kam", "ts": 1.0,
             "crop": crop, "bbox": (0, 0, 8, 8), "quality": 0.5}
    ctx = {"events": [ev_in]}
    m.process(ctx)
    sent = rm.sent[0]
    assert "crop" not in sent and isinstance(sent["jpeg"], bytes)
    assert "crop" in ctx["events"][0]   # orijinal event DEGISMEDI (kopya gonderildi)


def test_kamera_branch_override_eklenir():
    rm = _FakeRM()
    m = ReportingModule()
    cfg = {"reporting": {"branch_id": "G"},
           "cameras": [{"name": "Kam", "reporting": {"branch_id": "S2"}}]}
    m.setup(cfg, "Kam", {"report_manager": rm})
    m.process({"events": [{"type": "counting_crossing", "camera": "Kam", "ts": 1.0}]})
    assert rm.sent[0]["branch_id"] == "S2"
