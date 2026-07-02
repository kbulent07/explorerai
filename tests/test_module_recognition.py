# tests/test_module_recognition.py
from modules.recognition import RecognitionModule


class _Track:
    def __init__(self, tid):
        self.best_crop = f"crop{tid}"
        self.best_bbox = (0, 0, 10, 10)
        self.best_score = 0.9
        self.first_seen = 1.0
        self.last_seen = 2.0
        self.best_time = 1.5


def test_recognition_finished_icin_on_capture_cagirir():
    calls = []
    m = RecognitionModule()
    m.setup({}, "Kam", {"on_capture": lambda *a: calls.append(a)})
    m.process({"finished": [_Track(1), _Track(2)], "camera": "Kam"})
    assert len(calls) == 2
    assert calls[0][0] == "Kam" and calls[0][1] == "crop1"


def test_recognition_on_capture_yoksa_noop():
    m = RecognitionModule()
    m.setup({}, "Kam", {"on_capture": None})
    m.process({"finished": [_Track(1)], "camera": "Kam"})   # patlamamali
