# tests/test_name_match.py
# RecentFaceStore.name_for_embedding: gecis olayini kimlige (isim) baglar.

import numpy as np

from recent import RecentFaceStore


def _emb(seed):
    v = np.zeros(8, dtype=np.float32)
    v[seed % 8] = 1.0
    return v


def test_ayni_embedding_ismi_dondurur():
    s = RecentFaceStore(sim_threshold=0.4)
    eid, _ = s.add("Cam", (0, 0, 10, 10), b"x" * 50, 0.7, embedding=_emb(0))
    s.set_name(eid, "Ahmet")
    assert s.name_for_embedding(_emb(0)) == "Ahmet"


def test_farkli_embedding_none():
    s = RecentFaceStore(sim_threshold=0.4)
    eid, _ = s.add("Cam", (0, 0, 10, 10), b"x" * 50, 0.7, embedding=_emb(0))
    s.set_name(eid, "Ahmet")
    assert s.name_for_embedding(_emb(3)) is None   # dik vektor -> benzerlik 0


def test_otomatik_isim_de_dondurulur():
    s = RecentFaceStore(sim_threshold=0.4)
    s.add("Cam", (0, 0, 10, 10), b"x" * 50, 0.7, embedding=_emb(0))  # auto "Kisi 1"
    assert s.name_for_embedding(_emb(0)) == "Kisi 1"


def test_none_girdi_none_doner():
    s = RecentFaceStore()
    assert s.name_for_embedding(None) is None


def test_name_of():
    s = RecentFaceStore(sim_threshold=0.4)
    eid, _ = s.add("Cam", (0, 0, 10, 10), b"x" * 50, 0.7, embedding=_emb(0))
    assert s.name_of(eid) == "Kisi 1"       # otomatik ad
    s.set_name(eid, "Ali")
    assert s.name_of(eid) == "Ali"
    assert s.name_of(99999) is None


def test_get_or_create_akisi():
    # name_provider mantigi: eslesme yoksa ekle -> ad al; ayni emb tekrar -> ayni ad
    s = RecentFaceStore(sim_threshold=0.4)
    e = _emb(2)
    assert s.name_for_embedding(e) is None          # bos depoda esleme yok
    eid, _ = s.add("sayim", (0, 0, 40, 40), b"j" * 30, 0.5, embedding=e)
    nm = s.name_of(eid)
    assert nm == "Kisi 1"
    assert s.name_for_embedding(e) == nm            # sonraki gecis ayni ismi bulur
