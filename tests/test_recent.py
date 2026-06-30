# tests/test_recent.py
# -----------------------------------------------------------------------------
# RecentFaceStore birim testleri: embedding ile ayni-kisi birlestirme, farkli
# embedding -> ayri kimlik, konum yedegi (embedding yokken), bayt-butcesi
# tahliyesi ve drop_camera. numpy disinda agir bagimlilik yok.
# -----------------------------------------------------------------------------

import unittest

import numpy as np

from recent import RecentFaceStore


def _unit(vec):
    v = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


JPEG = b"x" * 1000  # sahte JPEG (boyut butce testi icin yeterli)


class TestEmbeddingMatching(unittest.TestCase):
    def test_same_embedding_merges(self):
        s = RecentFaceStore(sim_threshold=0.4)
        e = _unit([1, 0, 0])
        id1, new1 = s.add("Cam", (0, 0, 10, 10), JPEG, 0.5, ts=1.0, embedding=e)
        id2, new2 = s.add("Cam", (200, 200, 10, 10), JPEG, 0.6, ts=2.0, embedding=e)
        self.assertTrue(new1)
        self.assertFalse(new2)               # ayni kisi -> tek kayit
        self.assertEqual(id1, id2)
        self.assertEqual(s.stats()["count"], 1)

    def test_different_embedding_separate(self):
        s = RecentFaceStore(sim_threshold=0.4)
        s.add("Cam", (0, 0, 10, 10), JPEG, 0.5, ts=1.0, embedding=_unit([1, 0, 0]))
        _id, new = s.add("Cam", (0, 0, 10, 10), JPEG, 0.5, ts=2.0,
                         embedding=_unit([0, 1, 0]))  # dik vektor -> benzerlik 0
        self.assertTrue(new)                 # farkli kisi -> ayri kayit
        self.assertEqual(s.stats()["count"], 2)

    def test_location_fallback_without_embedding(self):
        s = RecentFaceStore()
        id1, _ = s.add("Cam", (0, 0, 20, 20), JPEG, 0.5, ts=1.0)        # embedding yok
        id2, new = s.add("Cam", (2, 2, 20, 20), JPEG, 0.6, ts=2.0)      # yakin konum
        self.assertEqual(id1, id2)           # konum yakin -> ayni kisi
        self.assertFalse(new)


class TestEvictionAndDrop(unittest.TestCase):
    def test_byte_budget_eviction(self):
        # 2 kayit sigacak butce: 3. kayit en eskiyi tahliye etmeli.
        # Ortogonal embedding'ler -> kosinus benzerligi 0 -> 3 AYRI kimlik
        # (benzer vektorler esik ustu kalir ve yanlislikla birlesirdi).
        s = RecentFaceStore(max_bytes=2500)
        embs = [_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])]
        for i, e in enumerate(embs):
            s.add("Cam", (i * 50, 0, 10, 10), JPEG, 0.5, ts=float(i), embedding=e)
        st = s.stats()
        self.assertLessEqual(st["bytes"], 2500)
        self.assertLessEqual(st["count"], 2)

    def test_drop_camera(self):
        s = RecentFaceStore(sim_threshold=0.4)
        s.add("A", (0, 0, 10, 10), JPEG, 0.5, ts=1.0, embedding=_unit([1, 0, 0]))
        s.add("B", (0, 0, 10, 10), JPEG, 0.5, ts=2.0, embedding=_unit([0, 1, 0]))
        dropped = s.drop_camera("A")
        self.assertEqual(dropped, 1)
        names = {e["camera"] for e in s.list_recent()}
        self.assertNotIn("A", names)

    def test_get_jpeg_roundtrip(self):
        s = RecentFaceStore()
        eid, _ = s.add("Cam", (0, 0, 10, 10), JPEG, 0.5, ts=1.0)
        self.assertEqual(s.get_jpeg(eid), JPEG)
        self.assertIsNone(s.get_jpeg(99999))


if __name__ == "__main__":
    unittest.main()
