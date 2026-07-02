# pipeline.py
# -----------------------------------------------------------------------------
# Config-driven islem hatti: kamera analiz/cikti asamalari (tanima, sayim, zoom,
# overlay) YAML'daki modul zinciri olarak calisir. Algilama cekirdegi (worker)
# ctx sozlugunu doldurur; buradaki moduller onu isler/cizer.
# -----------------------------------------------------------------------------

import importlib
import logging

log = logging.getLogger("aieye.pipeline")


class PipelineModule:
    """Islem hatti modulu tabani. Alt siniflar ihtiyac duydugu metodu uygular;
    hepsi opsiyonel (taban no-op saglar)."""

    def setup(self, config, camera, services):
        """Bir kez kurulum. config: tam config dict; camera: kamera adi;
        services: uygulama singleton'lari (on_capture, line_counter, ...)."""
        return None

    def process(self, ctx):
        """ANALIZ fazi: ctx'i gunceller (embedding, event...). Ciktiyi CIZMEZ."""
        return None

    def draw(self, ctx):
        """DISPLAY fazi: ctx['output'] uzerinde calisir (annotate/transform)."""
        return None

    def finalize(self):
        """Kapanista kaynak birak (thread, dedektor...)."""
        return None


def load_module(path, config, camera, services):
    """'paket.modul:SinifAdi' -> ornek. Yukleyip setup() cagirir. Hata olursa
    None doner (zincir kalanla devam etsin diye). Birkac constructor imzasi denenir."""
    if not isinstance(path, str) or ":" not in path:
        log.warning("Gecersiz modul yolu (paket.modul:Sinif bekleniyor): %r", path)
        return None
    mod_path, _, cls_name = path.partition(":")
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
    except Exception:
        log.warning("Modul yuklenemedi: %s (atlaniyor)", path, exc_info=True)
        return None
    # Birkac constructor imzasi dene: (config, camera), (config), ()
    inst = None
    for args in ((config, camera), (config,), ()):
        try:
            inst = cls(*args)
            break
        except TypeError:
            continue
        except Exception:
            log.warning("Modul kurucu hatasi: %s (atlaniyor)", path, exc_info=True)
            return None
    if inst is None:
        log.warning("Modul kurucusu hicbir imzayla kurulamadi: %s", path)
        return None
    try:
        inst.setup(config, camera, services)
    except Exception:
        log.warning("Modul setup hatasi: %s (atlaniyor)", path, exc_info=True)
        return None
    return inst


class Pipeline:
    """Modul zinciri. run(ctx): once tum process (analiz), sonra tum draw
    (display), liste sirasiyla. Bir modul patlarsa yakalanir + throttled loglanir;
    o kare icin o modul atlanir, zincir/canli akis kesilmez."""

    def __init__(self, modules):
        self.modules = list(modules)
        self._err_counts = {}   # (id(modul), faz) -> hata sayaci (log throttle)

    def _run_one(self, phase, fn, ctx):
        try:
            fn(ctx)
        except Exception:
            key = (id(fn.__self__), phase)
            n = self._err_counts.get(key, 0) + 1
            self._err_counts[key] = n
            if n == 1 or n % 50 == 0:
                log.warning("Modul %s.%s hatasi (#%d, kare atlaniyor)",
                            type(fn.__self__).__name__, phase, n, exc_info=True)

    def run(self, ctx):
        for m in self.modules:
            self._run_one("process", m.process, ctx)
        for m in self.modules:
            self._run_one("draw", m.draw, ctx)

    def finalize(self):
        for m in self.modules:
            try:
                m.finalize()
            except Exception:
                log.warning("Modul %s finalize hatasi", type(m).__name__,
                            exc_info=True)


# Varsayilan zincir: pipeline: tanimsizsa bugunku davranisi uretir.
def _default_chain(config, is_counting):
    chain = ["modules.recognition:RecognitionModule"]
    if is_counting:
        chain.append("modules.counting:CountingModule")
    if config.get("zoom_enabled", True):
        chain.append("modules.zoom:ZoomModule")
    if config.get("debug_overlay", True):
        chain.append("modules.overlay:OverlayModule")
    # Raporlama acik ise zincirin SONUNA ekle: process fazi liste sirasiyla
    # calisir -> uretici moduller (recognition/counting) event'leri once yazar.
    if (config.get("reporting") or {}).get("enabled", False):
        chain.append("modules.reporting:ReportingModule")
    return chain


def build_pipeline(config, camera, services, *, is_counting=False,
                   camera_pipeline=None):
    """Modul zincirini kur. Oncelik: camera_pipeline (kamera-bazli) > config['pipeline']
    (global) > varsayilan sentez. services: modullere setup ile verilecek singleton'lar."""
    if camera_pipeline is not None:
        paths = camera_pipeline
    elif isinstance(config.get("pipeline"), (list, tuple)):
        paths = list(config["pipeline"])
    else:
        paths = _default_chain(config, is_counting)
    modules = []
    for p in paths:
        m = load_module(p, config, camera, services)
        if m is not None:
            modules.append(m)
    return Pipeline(modules)
