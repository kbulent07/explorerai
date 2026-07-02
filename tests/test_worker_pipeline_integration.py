# tests/test_worker_pipeline_integration.py
# Varsayilan zincirle CameraWorker: pipeline kurulur, process() ctx doldurup
# zinciri calistirir, ciktiyi output_size'a resize eder. (Kamera/mediapipe gercek;
# detect gelmese de output ham hires'ten uretilir.)
import numpy as np

import worker as W


class _Cam:
    name = "Kam"
    connected = True

    def read_detect(self):
        return None, 0

    def read_hires(self):
        return np.zeros((120, 160, 3), np.uint8), 1


def test_pipeline_kurulur_ve_process_output_verir():
    captured = []
    cfg = {"detector_backend": "mediapipe", "zoom_enabled": False,
           "debug_overlay": False, "recognition_enabled": True,
           "output_size": [160, 120], "detect_interval": 1}
    w = W.CameraWorker(_Cam(), cfg, on_capture=lambda *a: captured.append(a))
    names = [type(m).__name__ for m in w._pipeline.modules]
    assert "RecognitionModule" in names        # varsayilan zincir kuruldu
    out = w.process()
    assert out is not None
    assert out.shape[:2] == (120, 160)         # output_size'a (w=160,h=120) resize


def test_set_zoom_zoommodule_varken_calisir():
    cfg = {"detector_backend": "mediapipe", "zoom_enabled": True,
           "debug_overlay": False, "output_size": [160, 120]}
    w = W.CameraWorker(_Cam(), cfg)
    w.set_zoom(False)   # ZoomModule'e delege; patlamamali
    zoom = [m for m in w._pipeline.modules if type(m).__name__ == "ZoomModule"]
    assert zoom and zoom[0].enabled is False


def test_report_manager_service_ve_modul(monkeypatch):
    class _Cam:
        name = "Kam"
        connected = True
        def read_detect(self):
            return None, 0
        def read_hires(self):
            return np.zeros((120, 160, 3), np.uint8), 1

    class _RM:
        def send(self, ev):
            return True

    cfg = {"detector_backend": "mediapipe", "zoom_enabled": False,
           "debug_overlay": False, "output_size": [160, 120],
           "detect_interval": 1, "reporting": {"enabled": True}}
    w = W.CameraWorker(_Cam(), cfg, report_manager=_RM())
    names = [type(m).__name__ for m in w._pipeline.modules]
    assert "ReportingModule" in names
