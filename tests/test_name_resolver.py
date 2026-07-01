# tests/test_name_resolver.py
# _NameResolver: gecis isim cozumunu ana donguden ayiran arka-plan thread'i.
# name_provider (ArcFace embed, agir) ayri thread'de calisir, cozulen isim
# CountingStore olayina set_name ile yazilir.

import time

from counting import CountingStore
from worker import _NameResolver


def test_name_resolver_olayi_asenkron_isimlendirir():
    store = CountingStore()
    eid = store.record("giris", ts=1.0)          # once isimsiz
    assert store.counts()["entered"][0]["name"] is None

    seen = {}

    def fake_provider(crop, camera, ts):
        seen["args"] = (crop, camera, ts)
        return "Zeynep"

    r = _NameResolver(fake_provider, store)
    try:
        r.submit(eid, b"cropdata", "Kamera1", 1.0)
        # arka plan thread'i isle -> olayin adi guncellenmeli
        for _ in range(50):
            if store.counts()["entered"][0]["name"] == "Zeynep":
                break
            time.sleep(0.01)
    finally:
        r.stop()

    assert store.counts()["entered"][0]["name"] == "Zeynep"
    assert seen["args"] == (b"cropdata", "Kamera1", 1.0)


def test_name_resolver_none_isimde_olayi_bozmaz():
    store = CountingStore()
    eid = store.record("cikis", ts=1.0)

    r = _NameResolver(lambda *a: None, store)   # isim cozulemez
    try:
        r.submit(eid, b"x", "Kam", 1.0)
        time.sleep(0.1)
    finally:
        r.stop()

    # None doner -> set_name cagrilmaz, olay isimsiz kalir (cokme yok)
    assert store.counts()["exited"][0]["name"] is None
