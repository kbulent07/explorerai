# config_store.py
# -----------------------------------------------------------------------------
# config.yaml icindeki KAMERA listesini guvenle okuyup yazar.
#
# Web ayarlar ekrani (webui.py) bu modulu kullanir. ruamel.yaml ile "round-trip"
# yapilir; boylece config.yaml icindeki aciklama satirlari (yorumlar) ve diger
# ayarlar KORUNUR, yalniz cameras listesi degissir.
#
# Hikvision RTSP URL'leri IP/kullanici/parola/kanal bilgisinden otomatik kurulur:
#   rtsp://KULLANICI:PAROLA@IP:PORT/Streaming/Channels/{kanal}{akis}
#     main (hires) -> {kanal}01    sub (detect) -> {kanal}02
# Parola/kullanicidaki ozel karakterler URL-encode edilir (@ -> %40 vb.).
# -----------------------------------------------------------------------------

import os
import re
import threading
import urllib.parse

from ruamel.yaml import YAML

import secrets_util

CONFIG_PATH = "config.yaml"

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.indent(mapping=2, sequence=2, offset=0)

_lock = threading.Lock()

# Salt-okuma onbellek: config.yaml her list_cameras() cagrisinda yeniden parse
# edilmesin (ruamel yavastir; web poll + her /live baglanisi bunu tetikliyordu).
# mtime degisince veya yazimdan sonra otomatik gecersiz olur.
_cache = {"mtime": None, "data": None}

# Basit host dogrulama: yalniz harf/rakam/nokta/tire/alt-cizgi (URL'ye ham
# gomulen ip alanindan bosluk/'/'/'@' vb. enjeksiyonu engelle).
_HOST_RE = re.compile(r"^[A-Za-z0-9._\-]+$")


def _load():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = _yaml.load(f)
    if data is None:
        data = {}
    if "cameras" not in data or data["cameras"] is None:
        data["cameras"] = []
    return data


def _load_cached():
    """Salt-okuma yol: config.yaml mtime degismediyse onbellekten dondur."""
    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        mtime = None
    if _cache["data"] is not None and _cache["mtime"] == mtime:
        return _cache["data"]
    data = _load()
    _cache["mtime"] = mtime
    _cache["data"] = data
    return data


def _save(data):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        _yaml.dump(data, f)
    # Yazimdan sonra onbellegi gecersiz kil -> sonraki okuma taze parse etsin.
    _cache["mtime"] = None
    _cache["data"] = None


def encrypt_existing():
    """Konfigdeki DUZ-METIN kamera parolalarini sifrele (varsa). Sifreleme devre
    disiysa (anahtar/cryptography yok) hicbir sey yapmaz. Degissen alan sayisini
    dondurur. Acilista bir kez cagrilir -> config'te artik duz parola kalmaz."""
    if not secrets_util.encryption_available():
        return 0
    with _lock:
        data = _load()
        changed = 0
        for cam in data.get("cameras", []) or []:
            if not cam:
                continue
            for key in ("detect_url", "hires_url"):
                url = cam.get(key)
                if not url:
                    continue
                enc = secrets_util.encrypt_url_password(url)  # zaten sifreliyse no-op
                if enc != url:
                    cam[key] = enc
                    changed += 1
        if changed:
            _save(data)
    return changed


def set_values(mapping):
    """Ust-seviye ayar anahtarlarini guncelle/ekle (yorumlar/diger ayarlar
    round-trip ile KORUNUR). Orn. counting_enabled/camera/line/swap."""
    with _lock:
        data = _load()
        for k, v in (mapping or {}).items():
            data[k] = v
        _save(data)


def load_full():
    """config.yaml'in TAMAMINI KILIT ALTINDA okuyup dondur (dict).

    Cagiranlar (webui: profil/dedektor degissince config'i tazeler) ham
    open()+parse yerine bunu kullanmali: _save ile ayni _lock paylasilir, boylece
    bir yazim (truncate+write) SIRASINDA yarim-dosya okunup parse hatasi olmaz.
    Ust-seviye dict; kameralar/quality_weights round-trip tiplerinde (dict/list
    gibi davranir) -> .get()/[] erisimi sorunsuz."""
    with _lock:
        return dict(_load())


def _validate_host(ip):
    """IP/hostname temel dogrulama (RTSP URL'sine ham gomulecek)."""
    ip = (ip or "").strip()
    if not ip or not _HOST_RE.match(ip):
        raise ValueError(f"Gecersiz IP/hostname: '{ip}'")
    return ip


def _validate_url(url, field="URL"):
    """RTSP/HTTP akis URL'si temel dogrulama (bosluk/yeni satir + sema)."""
    url = (url or "").strip()
    if any(c.isspace() for c in url):
        raise ValueError(f"{field} bosluk/yeni satir iceremez.")
    if "://" not in url:
        raise ValueError(f"{field} bir sema icermeli (orn. rtsp://...).")
    return url


def build_hik_url(ip, username, password, channel, stream, port=554):
    """Hikvision RTSP URL kur. stream: 'main' (01) veya 'sub' (02)."""
    ip = _validate_host(ip)
    suffix = "01" if stream == "main" else "02"
    user_enc = urllib.parse.quote(str(username), safe="")
    pass_enc = urllib.parse.quote(str(password), safe="")
    return (
        f"rtsp://{user_enc}:{pass_enc}@{ip}:{int(port)}"
        f"/Streaming/Channels/{int(channel)}{suffix}"
    )


def list_cameras():
    """Mevcut kameralari [(index, camera_dict), ...] olarak dondur."""
    with _lock:
        data = _load_cached()
        return list(enumerate(data.get("cameras", [])))


def _norm_role(role):
    """Kamera rolunu normalize et: 'giris' | 'cikis' | None."""
    r = (role or "").strip().lower()
    return r if r in ("giris", "cikis") else None


def camera_roles():
    """{kamera_adi: 'giris'|'cikis'} (yalniz rol atanmis kameralar)."""
    out = {}
    for _i, cam in list_cameras():
        r = _norm_role((cam or {}).get("role"))
        if r and cam.get("name"):
            out[cam["name"]] = r
    return out


def add_camera(name, detect_url, hires_url=None, role=None):
    """Tam URL'lerle yeni kamera ekle. hires_url None -> tek akis.
    role: 'giris' | 'cikis' | None (giris/cikis sayimi icin)."""
    name = (name or "").strip()
    detect_url = (detect_url or "").strip()
    if not name:
        raise ValueError("Kamera adi boss olamaz.")
    if not detect_url:
        raise ValueError("detect_url boss olamaz.")
    detect_url = _validate_url(detect_url, "detect_url")

    cam = {"name": name, "detect_url": secrets_util.encrypt_url_password(detect_url)}
    if hires_url and hires_url.strip():
        cam["hires_url"] = secrets_util.encrypt_url_password(
            _validate_url(hires_url, "hires_url"))
    role = _norm_role(role)
    if role:
        cam["role"] = role

    with _lock:
        data = _load()
        # Ayni isimde kamera varsa hata ver
        if any((c or {}).get("name") == name for c in data["cameras"]):
            raise ValueError(f"'{name}' adli kamera zaten var.")
        data["cameras"].append(cam)
        _save(data)
    return cam


def add_hik_camera(name, ip, username, password, channel=1, port=554,
                   single_stream=False, role=None):
    """Hikvision parametrelerinden URL'leri kurup kamera ekle.

    single_stream=True -> yalniz sub akis (detect = hires gibi davranir).
    role: 'giris' | 'cikis' | None.
    """
    name = (name or "").strip()
    ip = (ip or "").strip()
    if not name:
        raise ValueError("Kamera adi boss olamaz.")
    if not ip:
        raise ValueError("IP adresi boss olamaz.")

    detect_url = build_hik_url(ip, username, password, channel, "sub", port)
    hires_url = None
    if not single_stream:
        hires_url = build_hik_url(ip, username, password, channel, "main", port)
    return add_camera(name, detect_url, hires_url, role=role)


def update_camera(index, name, detect_url, hires_url=None, role=None):
    """index'teki kamerayi guncelle (ad + detect_url + opsiyonel hires_url + role).

    Saklanan temsil dogrudan duzenlenir (Hikvision ya da ozel URL fark etmez);
    boylece her kamera turu duzenlenebilir. Guncellenen dict'i dondurur (veya
    index gecersizse None). Yorumlar/diger ayarlar round-trip ile KORUNUR.
    """
    name = (name or "").strip()
    detect_url = (detect_url or "").strip()
    if not name:
        raise ValueError("Kamera adi boss olamaz.")
    if not detect_url:
        raise ValueError("detect_url boss olamaz.")
    detect_url = _validate_url(detect_url, "detect_url")

    with _lock:
        data = _load()
        cams = data["cameras"]
        if not (0 <= index < len(cams)):
            return None
        # Ad benzersizligi (duzenlenen kamera HARIC)
        for i, c in enumerate(cams):
            if i != index and (c or {}).get("name") == name:
                raise ValueError(f"'{name}' adli kamera zaten var.")
        cam = cams[index]
        if cam is None:
            cam = {}
            cams[index] = cam
        # Parola maskeli ('****') geldiyse mevcut (sifreli) parolayi koru; yeni
        # parola girildiyse sifrele. Boylece maskeli form parolayi bozmaz.
        old_detect = cam.get("detect_url")
        old_hires = cam.get("hires_url")
        cam["name"] = name
        cam["detect_url"] = secrets_util.merge_url_password(detect_url, old_detect)
        if hires_url and hires_url.strip():
            cam["hires_url"] = secrets_util.merge_url_password(
                _validate_url(hires_url, "hires_url"), old_hires)
        else:
            cam.pop("hires_url", None)   # bossaltilirsa tek akis moduna gec
        role = _norm_role(role)
        if role:
            cam["role"] = role
        else:
            cam.pop("role", None)        # rol secilmediyse kaldir
        _save(data)
        return dict(cam)


def delete_camera(index):
    """Verilen index'teki kamerayi sil. Silinen kamera dict'i dondur (veya None)."""
    with _lock:
        data = _load()
        cams = data["cameras"]
        if 0 <= index < len(cams):
            removed = cams.pop(index)
            _save(data)
            return removed
    return None
