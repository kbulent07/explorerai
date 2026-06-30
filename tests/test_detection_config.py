# resolve_detection_config + build_person_detector (model yok) testleri.
import pytest

from detection_backend import resolve_detection_config, build_person_detector


def test_defaults_when_empty():
    r = resolve_detection_config({})
    assert r["backend"] == "mediapipe"
    assert r["yolox_input_size"] == 416
    assert r["yolox_providers"] == ["CPUExecutionProvider"]
    assert 0.0 < r["yolox_confidence"] < 1.0


def test_invalid_backend_falls_back_to_mediapipe():
    r = resolve_detection_config({"detector_backend": "uydurma"})
    assert r["backend"] == "mediapipe"


def test_overrides_are_read():
    r = resolve_detection_config({
        "detector_backend": "yolox_person",
        "yolox_input_size": 640,
        "yolox_confidence": 0.5,
        "yolox_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "bytetrack_lost_buffer": 50,
    })
    assert r["backend"] == "yolox_person"
    assert r["yolox_input_size"] == 640
    assert r["yolox_confidence"] == 0.5
    assert r["yolox_providers"][0] == "CUDAExecutionProvider"
    assert r["bytetrack_lost_buffer"] == 50


def test_build_detector_returns_none_when_model_missing():
    r = resolve_detection_config({
        "detector_backend": "yolox_person",
        "yolox_model": "models/_yok_olan_model_.onnx",
    })
    # Model dosyasi yok -> None (worker mediapipe'e duser), istisna FIRLATMAZ
    assert build_person_detector(r) is None
