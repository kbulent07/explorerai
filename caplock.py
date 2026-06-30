# caplock.py
# -----------------------------------------------------------------------------
# Tek-ornek (single-instance) yakalama nobeti.
#
# main.py ve webui.py'nin HER IKISI de kamera akislarini acar. Ikisi ayni anda
# calisirsa kamera basina detect+hires RTSP akislari IKI kez acilir; cogu
# Hikvision eszamanli akis sayisini sinirladigindan akislar kopabilir/baglanamaz
# (sessiz operasyonel hata).
#
# Burada 127.0.0.1 uzerinde sabit bir porta baglanarak isletim sisteminden
# "tek yakalayici" garantisi alinir: ikinci surec o porta baglanamaz -> UYARIR.
# Bloke ETMEZ; kullanici bilinerek yine de devam edebilir. Soket surec sonlaninca
# isletim sistemi tarafindan otomatik birakilir -> bayat kilit dosyasi sorunu yok.
# -----------------------------------------------------------------------------

import logging
import socket

log = logging.getLogger("facezoom.caplock")

# Surec boyunca acik tutulan nobet soketi (port mesgul kalsin diye).
_guard_sock = None


def acquire(port=47632):
    """Yakalama nobetini al. Baska bir yakalayici varsa False dondur + uyar."""
    global _guard_sock
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        s.listen(1)
        _guard_sock = s   # GC edilmesin: port surec boyunca mesgul kalir
        return True
    except OSError:
        s.close()
        log.warning(
            "Baska bir FaceZoom yakalama surecinin (main.py veya webui.py) zaten "
            "calistigi gorunuyor. Iki surec ayni kameralari ayni anda acarsa RTSP "
            "akislari kopabilir/baglanamayabilir. Yalniz BIRINI calistirin."
        )
        return False
