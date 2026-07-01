# perf.py
# -----------------------------------------------------------------------------
# Islemci profilleri. Tek bir `cpu_profile` ayariyla bir grup islem-etkili
# secenegi topluca degistirir; kullanici tek tek ayar girmek zorunda kalmaz.
#
#   cpu_profile: normal   -> config.yaml'daki degerler aynen kullanilir (varsayilan)
#   cpu_profile: low      -> dussuk-CPU degerleri UYGULANIR (ezer)
#   cpu_profile: high     -> yuksek-DOGRULUK degerleri UYGULANIR (ezer) - GPU onerilir
#
# "low"/"high" secildiginde ilgili anahtarlar profil tarafindan belirlenir (elle
# degerler yok sayilir). Ince ayar isteyen 'normal'e alip anahtarlari kendisi girer.
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

# Yuksek-dogruluk profilinin ezdigi ayarlar (GPU onerilir; CPU'da AGIRDIR):
# Algilamayi TAM cozunurlukte ve HER karede yaparak kucuk/uzak/egik yuzleri
# daha iyi yakalar -> tanima icin daha kaliteli kirpinti besler.
# NOT: recognition_det_size yine PROFILE DAHIL DEGIL; yuz tanima dogrulugunu
# ayrica artirmak icin arayuzden (Yuz Tanima Cozunurlugu) 640 secin veya GPU
# kurulum scripti bunu otomatik yapar.
HIGH_ACCURACY = {
    "detect_on_hires": True,       # algilamayi yuksek coz. akista yap (kucuk yuzler)
    "detect_downscale": 1.0,       # kucultme YOK -> uzak/kucuk yuzler kacmaz
    "preview_fps": 15,             # daha akici isleme/takip
    "detect_interval": 1,          # her kare algila (en iyi takip surekliligi)
    "zoom_enabled": True,          # yuze-odakli pan-zoom acik
}

# Profil adi -> ezilen ayarlar tablosu ('normal' hicbir seyi ezmez)
_PROFILE_OVERRIDES = {
    "low": LOW_CPU,
    "high": HIGH_ACCURACY,
}

VALID_PROFILES = ("normal", "low", "high")


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
    'low'/'high' ise ilgili anahtarlari ezer; 'normal' ise dokunmaz."""
    if config is None:
        return config
    overrides = _PROFILE_OVERRIDES.get(resolve_profile(config))
    if overrides:
        for k, v in overrides.items():
            config[k] = list(v) if isinstance(v, list) else v
    return config
