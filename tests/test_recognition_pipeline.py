# tests/test_recognition_pipeline.py
# RecognitionPipeline drop/degrade davranisi:
#  - taniyici YUKLENEMEZSE (embed exception) -> yuz DUSURULMEZ, embed'siz saklanir
#    (galeri bos kalmasin; kirpinti zaten MediaPipe yuz tespitinden gecti)
#  - taniyici CALISIR ama yuz DOGRULAMAZSA (None) + require_face -> DUSURULUR
#  - embedding donerse -> saklanir

import time

import numpy as np

from recent import RecentFaceStore
from recognition import RecognitionPipeline

_CROP = np.zeros((40, 40, 3), dtype=np.uint8)


def _wait_processed(pipe, store, want, timeout=2.0):
    """Kuyruk bosalana + beklenen sayiya ulasana kadar bekle (veya timeout)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if store.stats()["count"] == want and len(pipe._dq) == 0:
            return
        time.sleep(0.02)


class _Raise:
    def embed(self, crop):
        raise RuntimeError("model yok / CUDA hatasi")


class _None:
    def embed(self, crop):
        return None


class _Emb:
    def embed(self, crop):
        v = np.ones(512, dtype=np.float32)
        return v / np.linalg.norm(v)


def test_taniyici_yuklenemezse_embedsiz_saklar():
    store = RecentFaceStore()
    p = RecognitionPipeline(_Raise(), store, require_face=True).start()
    try:
        p.submit("Kam", _CROP, (0, 0, 40, 40), 0.9, ts=1.0)
        _wait_processed(p, store, 1)
    finally:
        p.stop()
    assert store.stats()["count"] == 1   # DUSURULMEDI (dayanikli degrade)
    assert p._load_failed is True


def test_yuz_dogrulanmazsa_require_face_dusurur():
    store = RecentFaceStore()
    p = RecognitionPipeline(_None(), store, require_face=True).start()
    try:
        p.submit("Kam", _CROP, (0, 0, 40, 40), 0.9, ts=1.0)
        time.sleep(0.3)   # islenmesi icin sure ver
    finally:
        p.stop()
    assert store.stats()["count"] == 0   # DUSURULDU
    assert p._dropped >= 1


def test_require_face_false_ise_none_de_saklar():
    store = RecentFaceStore()
    p = RecognitionPipeline(_None(), store, require_face=False).start()
    try:
        p.submit("Kam", _CROP, (0, 0, 40, 40), 0.9, ts=1.0)
        _wait_processed(p, store, 1)
    finally:
        p.stop()
    assert store.stats()["count"] == 1


def test_embedding_donerse_saklar():
    store = RecentFaceStore()
    p = RecognitionPipeline(_Emb(), store, require_face=True).start()
    try:
        p.submit("Kam", _CROP, (0, 0, 40, 40), 0.9, ts=1.0)
        _wait_processed(p, store, 1)
    finally:
        p.stop()
    assert store.stats()["count"] == 1
