# tests/test_worker_backend_selection.py
# -----------------------------------------------------------------------------
# Kamera veya model OLMADAN CameraWorker arka uc secimini dogrular (headless).
#
# FaceTracker mediapipe yuklediginden (model dosyasini acar) kamerali ortamda
# agir olabilir; bu yuzden worker.FaceTracker monkeypatch ile stub'laniyor.
# Bu yaklasim: (1) testi hafif/hizli tutar, (2) mediapipe kurulumuna bagimli
# degildir, (3) gercek CameraWorker __init__ dallanma mantigi DEGISTIRILMEDEN
# test edilir (sadece FaceTracker instansiasyonu yerine stub geciyor).
# -----------------------------------------------------------------------------

import pytest
import numpy as np

import worker  # test edilen modul


class _FakeTracker:
    """FaceTracker stub: sadece kur/kapat, detect cagrilmaz (process() cagirilmiyor)."""
    def __init__(self, **kwargs):
        pass
    def detect(self, frame):
        return []
    def close(self):
        pass


class _FakeCamera:
    """CameraWorker.__init__'in ihtiyac duydugu camera arayuzu (adi ve connected)."""
    name = "test_cam"
    connected = True


@pytest.fixture(autouse=True)
def patch_face_tracker(monkeypatch):
    """Her testte FaceTracker'i stub ile degistir."""
    monkeypatch.setattr(worker, "FaceTracker", _FakeTracker)


# --------------------------------------------------------------------------- #
# Test 1: Varsayilan config -> mediapipe yolu                                 #
# --------------------------------------------------------------------------- #

def test_default_config_uses_mediapipe():
    """Configde detector_backend yoksa backend mediapipe, manager FaceTrackerManager olmali."""
    from tracking import FaceTrackerManager
    config = {}
    w = worker.CameraWorker(_FakeCamera(), config)
    assert w.backend == "mediapipe"
    assert isinstance(w.manager, FaceTrackerManager)


# --------------------------------------------------------------------------- #
# Test 2: yolox_person + var olmayan model -> mediapipe fallback              #
# --------------------------------------------------------------------------- #

def test_yolox_missing_model_falls_back_to_mediapipe():
    """yolox_person istendi ama model yoksa backend mediapipe'e dusmeli."""
    from tracking import FaceTrackerManager
    config = {
        "detector_backend": "yolox_person",
        "yolox_model": "models/_yok_olan_model_test_.onnx",
    }
    w = worker.CameraWorker(_FakeCamera(), config)
    assert w.backend == "mediapipe"
    assert isinstance(w.manager, FaceTrackerManager)
    # person_detector None olmali (model yoktu)
    assert w._person_detector is None


# --------------------------------------------------------------------------- #
# Test 3: Gecersiz backend -> mediapipe fallback                              #
# --------------------------------------------------------------------------- #

def test_invalid_backend_falls_back_to_mediapipe():
    """Tanimsiz bir detector_backend degerinde de mediapipe secilmeli."""
    from tracking import FaceTrackerManager
    config = {"detector_backend": "uydurma_backend"}
    w = worker.CameraWorker(_FakeCamera(), config)
    assert w.backend == "mediapipe"
    assert isinstance(w.manager, FaceTrackerManager)


# --------------------------------------------------------------------------- #
# Test 4: mediapipe config -> _center_dist_factor okunuyor                   #
# --------------------------------------------------------------------------- #

def test_center_dist_factor_is_read():
    """track_center_dist_factor config degerinden okunmali."""
    config = {"track_center_dist_factor": 0.35}
    w = worker.CameraWorker(_FakeCamera(), config)
    assert w._center_dist_factor == pytest.approx(0.35)
