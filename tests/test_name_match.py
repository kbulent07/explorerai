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
