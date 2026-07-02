# pipeline.py
# -----------------------------------------------------------------------------
# Config-driven islem hatti: kamera analiz/cikti asamalari (tanima, sayim, zoom,
# overlay) YAML'daki modul zinciri olarak calisir. Algilama cekirdegi (worker)
# ctx sozlugunu doldurur; buradaki moduller onu isler/cizer.
# -----------------------------------------------------------------------------

import importlib
import logging

log = logging.getLogger("facezoom.pipeline")


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
