# tests/test_perf.py
# CPU profili: 'low' dussuk-CPU ayarlarini ezer; 'normal' dokunmaz.

import perf


def test_low_profil_ezer():
    cfg = {"cpu_profile": "low", "preview_fps": 30, "zoom_enabled": True,
           "detect_on_hires": True, "output_size": [1920, 1080]}
    out = perf.apply_cpu_profile(cfg)
    assert out["preview_fps"] == perf.LOW_CPU["preview_fps"]
    assert out["zoom_enabled"] is False
    assert out["detect_on_hires"] is False
    assert out["output_size"] == perf.LOW_CPU["output_size"]


def test_normal_profil_dokunmaz():
    cfg = {"cpu_profile": "normal", "preview_fps": 30, "zoom_enabled": True}
    out = perf.apply_cpu_profile(cfg)
    assert out["preview_fps"] == 30
    assert out["zoom_enabled"] is True


def test_varsayilan_normal():
    assert perf.resolve_profile({}) == "normal"
    cfg = {"preview_fps": 30}
    assert perf.apply_cpu_profile(cfg)["preview_fps"] == 30   # profilsiz -> dokunmaz


def test_gecersiz_profil_normal_sayilir():
    assert perf.resolve_profile({"cpu_profile": "sacma"}) == "normal"


def test_onnx_providers_cpu_varsayilan():
    assert perf.onnx_providers({}) == ["CPUExecutionProvider"]
    assert perf.onnx_providers({"compute_device": "cpu"}) == ["CPUExecutionProvider"]


def test_onnx_providers_gpu_cuda_once():
    p = perf.onnx_providers({"compute_device": "gpu"})
    assert p[0] == "CUDAExecutionProvider" and "CPUExecutionProvider" in p


def test_onnx_providers_gecersiz_cpu():
    assert perf.onnx_providers({"compute_device": "sacma"}) == ["CPUExecutionProvider"]


def test_output_size_kopya_referans_paylasmaz():
    cfg = {"cpu_profile": "low"}
    a = perf.apply_cpu_profile(cfg)["output_size"]
    a.append(999)                       # dis mutasyon
    assert 999 not in perf.LOW_CPU["output_size"]   # sablon bozulmadi
