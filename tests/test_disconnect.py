# tests/test_disconnect.py
# LiveManager baglanti-kes/yeniden-bagla durum mantigi.

from live import LiveManager


def test_disconnect_otomatik_baslatmayi_engeller():
    lm = LiveManager(config={})
    assert not lm.is_disconnected("Cam")
    lm.disconnect("Cam")
    assert lm.is_disconnected("Cam")
    # 'kesildi' isaretliyken ensure OTOMATIK baslatmaz -> None
    assert lm.ensure("Cam") is None


def test_reconnect_isareti_temizler():
    lm = LiveManager(config={})
    lm.disconnect("Cam")
    lm.reconnect("Cam")                    # config'te 'Cam' yok -> baslamaz ama
    assert not lm.is_disconnected("Cam")   # isaret temizlenir


def test_disconnected_names_seti():
    lm = LiveManager(config={})
    lm.disconnect("A")
    lm.disconnect("B")
    assert lm.disconnected_names() == {"A", "B"}
    lm.reconnect("A")
    assert lm.disconnected_names() == {"B"}
