# tests/test_module_counting.py
import numpy as np

from modules.counting import CountingModule
from counting import LineCrossingCounter, CountingStore


class _FakeMgr:
    tracks = {}


def _ctx(tracks):
    return {"tracks": tracks, "detect_dims": (200, 200), "tid_face": {},
            "scale": (1.0, 1.0), "hires_frame": np.zeros((200, 200, 3), np.uint8),
            "now": 1.0, "camera": "Kam"}


def test_counting_gecis_kaydeder():
    lc = LineCrossingCounter(line=(0.0, 0.5, 1.0, 0.5))
    store = CountingStore()
    m = CountingModule()
    m.setup({}, "Kam", {"line_counter": lc, "counting_store": store,
                        "name_resolver": None, "manager": _FakeMgr()})
    m.process(_ctx([(1, (90, 40, 20, 20))]))    # ust taraf
    m.process(_ctx([(1, (90, 140, 20, 20))]))   # cizgiyi gecti
    c = store.counts()
    assert c["in"] + c["out"] == 1


def test_counting_sayac_yoksa_noop():
    m = CountingModule()
    m.setup({}, "Kam", {"line_counter": None, "counting_store": None,
                        "name_resolver": None, "manager": _FakeMgr()})
    m.process(_ctx([(1, (90, 140, 20, 20))]))   # patlamamali


def test_counting_event_uretir():
    lc = LineCrossingCounter(line=(0.0, 0.5, 1.0, 0.5))
    store = CountingStore()
    m = CountingModule()
    m.setup({}, "Kam", {"line_counter": lc, "counting_store": store,
                        "name_resolver": None, "manager": _FakeMgr()})
    c1 = _ctx([(1, (90, 40, 20, 20))])
    m.process(c1)
    assert not c1.get("events")            # gecis yok -> event yok
    c2 = _ctx([(1, (90, 140, 20, 20))])
    m.process(c2)
    evs = c2.get("events") or []
    assert len(evs) == 1
    ev = evs[0]
    assert ev["type"] == "counting_crossing"
    assert ev["camera"] == "Kam"
    assert ev["direction"] in ("in", "out")
    assert ev["name"] is None
