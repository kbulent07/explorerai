# perf.py
# -----------------------------------------------------------------------------
# Islemci (CPU) profilleri. Tek bir `cpu_profile` ayariyla bir grup CPU-etkili
# secenegi topluca degistirir; kullanici tek tek ayar girmek zorunda kalmaz.
#
#   cpu_profile: normal   -> config.yaml'daki degerler aynen kullanilir (varsayilan)
#   cpu_profile: low      -> asagidaki dussuk-CPU degerleri UYGULANIR (ezer)
#
# "low" secildiginde bu anahtarlar profil tarafindan belirlenir (elle degerler
# yok sayilir). Ince ayar isteyen 'normal'e alip anahtarlari kendisi girer.
# -----------------------------------------------------------------------------

# Dussuk-CPU profilinin ezdigi ayarlar:
# NOT: recognition_det_size PROFILE DAHIL DEGIL -> arayuzden ayri secilir
# (kullanici secimi profil tarafindan ezilmesin).
LOW_CPU = {
    "detect_on_hires": False,      # 1080p'yi algilama icin decode/resize etme
    "detect_downscale": 0.35,      # (detect_on_hires acik kalirsa) daha kucuk
    "preview_fps": 6,              # onizleme/isleme hizi ust siniri
    "detect_interval": 4,          # kac karede bir algilama
    "output_size": [960, 540],     # cikti cozunurlugu (resize+encode ucuzlar)
    "zoom_enabled": False,         # dijital pan-zoom (resize) kapali
}

VALID_PROFILES = ("normal", "low")


def onnx_providers(config):
    """compute_device'a gore onnxruntime sağlayici listesi (YOLOX + insightface
    ORTAK kullanir). 'gpu' -> once CUDA, yoksa CPU'ya duser. 'cpu' -> yalniz CPU.
    GPU icin imajda onnxruntime-gpu + CUDA kutuphaneleri gerekir (bkz. Dockerfile.gpu)."""
    dev = str((config or {}).get("compute_device", "cpu")).strip().lower()
    if dev == "gpu":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def resolve_profile(config):
    prof = str((config or {}).get("cpu_profile", "normal")).strip().lower()
    return prof if prof in VALID_PROFILES else "normal"


def apply_cpu_profile(config):
    """config sozlugune aktif profili uygula (yerinde). config'i geri dondurur.
    'low' ise LOW_CPU anahtarlarini ezer; 'normal' ise dokunmaz."""
    if config is None:
        return config
    if resolve_profile(config) == "low":
        for k, v in LOW_CPU.items():
            config[k] = list(v) if isinstance(v, list) else v
    return config
