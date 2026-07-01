# secrets_util.py
# -----------------------------------------------------------------------------
# Kamera RTSP parolalarini KONFIGDE sifreli tutmak icin yardimcilar.
#
# RTSP baglantisi GERCEK parolayi ister -> geri-donusturulebilir sifreleme
# (Fernet/AES) kullaniriz; tek yonlu hash OLMAZ. Anahtar:
#   1) FACEZOOM_SECRET_KEY ortam degiskeni (Docker'da onerilen), veya
#   2) .facezoom.key dosyasi (yoksa uretilir; gitignore'lu, tasinabilir).
#
# Sifreli parola URL icinde "enc$<token>" olarak gomulur:
#   rtsp://admin:enc$gAAAA...@10.0.0.1:554/...
# VideoCapture'a verilmeden ONCE cozulur (camera._open). Arayuzde MASKE gosterilir.
#
# GERI UYUM: Duz-metin parolalar (enc$ yok) oldugu gibi calisir; cryptography
# yoksa ya da anahtar cozulemezse sifreleme sessizce DEVRE DISI kalir (duz metin)
# -> kameralar asla kirilmaz.
# -----------------------------------------------------------------------------

import base64
import hashlib
import logging
import os
import urllib.parse

log = logging.getLogger("facezoom.secrets")

_MARK = "enc$"
_cache = {"loaded": False, "fernet": None}


def _derive_key(passphrase):
    """Rastgele bir parolayi gecerli bir 32-byte Fernet anahtarina indir."""
    return base64.urlsafe_b64encode(hashlib.sha256(passphrase.encode()).digest())


def _is_fernet_key(s):
    try:
        return len(base64.urlsafe_b64decode(s.encode())) == 32
    except Exception:
        return False


def _fernet():
    """Fernet ornegini (tembel) yukle. Sifreleme mumkun degilse None."""
    if _cache["loaded"]:
        return _cache["fernet"]
    _cache["loaded"] = True
    try:
        from cryptography.fernet import Fernet
    except Exception:
        log.warning("cryptography kurulu degil -> parolalar DUZ METIN saklanir. "
                    "Sifreleme icin requirements'taki cryptography'yi kurun.")
        _cache["fernet"] = None
        return None

    # Anahtar YALNIZ ortam degiskeninden gelir (kullanici yonetir). Otomatik
    # keyfile URETILMEZ: aksi halde Docker restart'ta anahtar degisip parolalar
    # cozulemez ve testlerde davranis sessizce degisirdi. Opsiyonel: FACEZOOM_KEY_FILE
    # ile MEVCUT bir anahtar dosyasi gosterilebilir (uretmez, sadece okur).
    secret = os.environ.get("FACEZOOM_SECRET_KEY")
    if not secret:
        keyfile = os.environ.get("FACEZOOM_KEY_FILE")
        if keyfile and os.path.exists(keyfile):
            try:
                with open(keyfile, "r", encoding="utf-8") as f:
                    secret = f.read().strip()
            except OSError:
                secret = None
    if not secret:
        log.info("FACEZOOM_SECRET_KEY yok -> kamera parolalari DUZ METIN saklanir "
                 "(arayuzde maskelenir). Sifreleme icin bu ortam degiskenini verin.")
        _cache["fernet"] = None
        return None

    try:
        key = secret if _is_fernet_key(secret) else _derive_key(secret)
        _cache["fernet"] = Fernet(key)
    except Exception:
        try:
            _cache["fernet"] = Fernet(_derive_key(secret))
        except Exception:
            log.error("Fernet anahtari kurulamadi -> duz metin")
            _cache["fernet"] = None
    return _cache["fernet"]


def encryption_available():
    return _fernet() is not None


def is_encrypted(token):
    return isinstance(token, str) and token.startswith(_MARK)


def encrypt(plaintext):
    """Duz metni 'enc$<token>' yap. Sifreleme yoksa duz metni oldugu gibi dondur."""
    if plaintext is None:
        plaintext = ""
    if is_encrypted(plaintext):
        return plaintext
    f = _fernet()
    if f is None:
        return plaintext
    return _MARK + f.encrypt(plaintext.encode()).decode()


def decrypt(token):
    """'enc$<token>' -> duz metin. Sifreli degilse/cozulemezse oldugu gibi dondur."""
    if not is_encrypted(token):
        return token
    f = _fernet()
    if f is None:
        return token
    try:
        return f.decrypt(token[len(_MARK):].encode()).decode()
    except Exception:
        log.error("Parola cozulemedi (anahtar degismis/kayip olabilir)")
        return token


# ---- RTSP URL parola islemleri --------------------------------------------

def _split_url(url):
    """rtsp://user:pass@host/path -> (scheme, user, pass, host_and_path) veya None.
    Not: config'te parola URL-encode'ludur -> pass icinde ham ':' / '@' yok."""
    if not isinstance(url, str) or "://" not in url:
        return None
    scheme, rest = url.split("://", 1)
    if "@" not in rest:
        return None
    creds, host = rest.rsplit("@", 1)
    if ":" not in creds:
        return None
    user, pw = creds.split(":", 1)
    return scheme, user, pw, host


def encrypt_url_password(url):
    """URL icindeki parolayi sifrele (enc$...). Zaten sifreli/cozulemezse dokunma."""
    parts = _split_url(url)
    if not parts:
        return url
    scheme, user, pw, host = parts
    if is_encrypted(pw):
        return url
    real = urllib.parse.unquote(pw)   # config'te URL-encoded
    enc = encrypt(real)
    if enc == real:                   # sifreleme devre disi -> dokunma
        return url
    return f"{scheme}://{user}:{enc}@{host}"


def decrypt_url_password(url):
    """URL icindeki 'enc$...' parolayi cozup URL-encode ederek geri koy
    (VideoCapture icin). Sifreli degilse oldugu gibi dondur."""
    parts = _split_url(url)
    if not parts:
        return url
    scheme, user, pw, host = parts
    if not is_encrypted(pw):
        return url
    real = decrypt(pw)
    return f"{scheme}://{user}:{urllib.parse.quote(real, safe='')}@{host}"


def password_encrypted_but_unresolved(url):
    """URL parolasi SIFRELI (enc$) ama COZULEMIYOR mu? (anahtar eksik/yanlis).

    True doner -> kamera bu parolayla baglanamaz (enc$ string'i ham parola gibi
    gonderilir, kimlik dogrulama basarisiz). En sik sebep: config.yaml baska bir
    PC'ye tasindi ama .env (FACEZOOM_SECRET_KEY) tasinmadi ya da setup yeni bir
    anahtar uretti. Cagiran (camera / web ayarlar) net uyari verebilir."""
    parts = _split_url(url)
    if not parts:
        return False
    pw = parts[2]
    if not is_encrypted(pw):
        return False
    # Cozulemezse decrypt() token'i AYNEN dondurur (fernet None ya da InvalidToken).
    return decrypt(pw) == pw


def mask_url_password(url):
    """Arayuzde gostermek icin parolayi '****' ile maskele."""
    parts = _split_url(url)
    if not parts:
        return url
    scheme, user, pw, host = parts
    return f"{scheme}://{user}:****@{host}"


MASK = "****"


def merge_url_password(submitted_url, existing_url):
    """Duzenleme formundan gelen URL parolasi hala '****' ise (kullanici
    degistirmedi) mevcut (sifreli) parolayi koru; degistiyse yeni parolayi
    sifrele. Boylece maskeli form kaydedilince parola bozulmaz."""
    parts = _split_url(submitted_url)
    if not parts:
        return submitted_url
    scheme, user, pw, host = parts
    if pw == MASK:
        old = _split_url(existing_url or "")
        if old:
            return f"{scheme}://{user}:{old[2]}@{host}"   # eski (sifreli) parolayi kullan
        return submitted_url
    return encrypt_url_password(submitted_url)
