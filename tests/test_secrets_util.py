# tests/test_secrets_util.py
# Parola sifreleme + maskeleme + URL islemleri.

import os

import secrets_util as s


def _reset():
    s._cache["loaded"] = False
    s._cache["fernet"] = None


def _with_key():
    os.environ["FACEZOOM_SECRET_KEY"] = "test-secret-key-123"
    _reset()


def _without_key():
    os.environ.pop("FACEZOOM_SECRET_KEY", None)
    os.environ.pop("FACEZOOM_KEY_FILE", None)
    _reset()


# --- anahtar VARKEN: gercek sifreleme ---

def test_roundtrip_encrypt_decrypt():
    _with_key()
    p = "p@ss:w/rd .com"
    enc = s.encrypt(p)
    assert s.is_encrypted(enc) and p not in enc
    assert s.decrypt(enc) == p


def test_url_password_encrypt_gizler_ve_geri_cozer():
    _with_key()
    u = "rtsp://admin:mypass123@10.0.0.1:554/Streaming/Channels/102"
    enc = s.encrypt_url_password(u)
    assert "enc$" in enc and "mypass123" not in enc
    assert s.decrypt_url_password(enc) == u


def test_mask_parolayi_gizler():
    _with_key()
    u = "rtsp://admin:mypass123@10.0.0.1:554/x"
    assert s.mask_url_password(u) == "rtsp://admin:****@10.0.0.1:554/x"
    enc = s.encrypt_url_password(u)
    assert s.mask_url_password(enc) == "rtsp://admin:****@10.0.0.1:554/x"


def test_merge_unchanged_eski_parolayi_korur():
    _with_key()
    u = "rtsp://admin:mypass123@10.0.0.1:554/x"
    enc = s.encrypt_url_password(u)
    merged = s.merge_url_password("rtsp://admin:****@10.0.0.1:554/x", enc)
    assert s.decrypt_url_password(merged) == u   # eski parola korundu


def test_merge_yeni_parola_sifrelenir():
    _with_key()
    old = s.encrypt_url_password("rtsp://admin:old@10.0.0.1:554/x")
    merged = s.merge_url_password("rtsp://admin:yenipar@10.0.0.1:554/x", old)
    assert "yenipar" not in merged and "enc$" in merged
    assert s.decrypt_url_password(merged) == "rtsp://admin:yenipar@10.0.0.1:554/x"


# --- anahtar YOKKEN: duz metin (geri uyum) ---

def test_anahtarsiz_duz_metin_gecer():
    _without_key()
    assert not s.encryption_available()
    u = "rtsp://admin:plain123@10.0.0.1:554/x"
    assert s.encrypt_url_password(u) == u          # dokunmaz
    assert s.decrypt_url_password(u) == u
    # maskeleme yine calisir (sifre gorunmesin)
    assert s.mask_url_password(u) == "rtsp://admin:****@10.0.0.1:554/x"


def test_kimliksiz_url_dokunulmaz():
    _without_key()
    u = "rtsp://10.0.0.1:554/x"   # user:pass yok
    assert s.mask_url_password(u) == u
    assert s.encrypt_url_password(u) == u
