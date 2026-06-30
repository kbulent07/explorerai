# tests/test_tracking.py
# -----------------------------------------------------------------------------
# tracking.py birim testleri: IoU, kalite skoru ve BELIRLEYICI track eslessmesi
# (B1 duzeltmesi). mediapipe/insightface import etmez -> hizli ve bagimsiz calisir.
#
# Calistir:  venv\Scripts\python -m unittest discover -s tests   (veya pytest)
# -----------------------------------------------------------------------------

import unittest

import numpy as np

from tracking import _iou, compute_quality, FaceTrackerManager, Track


class TestIoU(unittest.TestCase):
    def test_identical_boxes(self):
        self.assertAlmostEqual(_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint_boxes(self):
        self.assertEqual(_iou((0, 0, 10, 10), (100, 100, 10, 10)), 0.0)

    def test_half_overlap(self):
        # (0..10) ile (5..15): kesisim 5x10=50, birlesim 100+100-50=150
        self.assertAlmostEqual(_iou((0, 0, 10, 10), (5, 0, 10, 10)), 50 / 150)


class TestComputeQuality(unittest.TestCase):
    def test_empty_crop_is_zero(self):
        face = {"bbox": (0, 0, 10, 10), "confidence": 0.9, "keypoints": {}}
        self.assertEqual(compute_quality(face, None, 1000.0, {}), 0.0)
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(compute_quality(face, empty, 1000.0, {}), 0.0)

    def test_returns_bounded_float(self):
        face = {"bbox": (0, 0, 40, 40), "confidence": 0.8, "keypoints": {}}
        crop = np.full((40, 40, 3), 120, dtype=np.uint8)
        score = compute_quality(face, crop, 40 * 40 * 4.0,
                                {"sharpness": 0.4, "size": 0.2, "frontality": 0.15,
                                 "exposure": 0.1, "confidence": 0.1})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.5)  # agirliklar ~1.0; ust sinir genis tutuldu


class TestDeterministicMatching(unittest.TestCase):
    """B1: yalniz mesafeyle eslesen adaylar arasinda EN YAKIN deterministik
    olarak secilmeli (eskiden set gezinme sirasina gore belirsizdi)."""

    def _two_track_manager(self):
        m = FaceTrackerManager(iou_threshold=0.3, max_center_dist=200.0,
                               track_timeout=2.0)
        m.tracks = {10: Track(10, (0, 0, 10, 10), 0.0),
                    20: Track(20, (100, 0, 10, 10), 0.0)}
        m._next_id = 100
        return m

    def test_picks_nearest_when_iou_zero(self):
        # det merkezi (65,5): track20 (105,5) -> 40px; track10 (5,5) -> 60px
        det = (60, 0, 10, 10)
        for _ in range(20):  # tekrar -> her seferinde AYNI sonuc (belirleyici)
            m = self._two_track_manager()
            assign = m.update([det], now=0.1)
            self.assertEqual(assign, [(20, det)])

    def test_prefers_high_iou_over_distance(self):
        m = self._two_track_manager()
        det = (2, 2, 10, 10)  # track10 ile yuksek IoU
        assign = m.update([det], now=0.1)
        self.assertEqual(assign[0][0], 10)

    def test_far_detection_opens_new_track(self):
        m = FaceTrackerManager(iou_threshold=0.3, max_center_dist=10.0,
                               track_timeout=2.0)
        a1 = m.update([(0, 0, 10, 10)], now=0.0)
        a2 = m.update([(500, 500, 10, 10)], now=0.1)
        self.assertNotEqual(a1[0][0], a2[0][0])  # yeni kimlik acilmali

    def test_timeout_finalizes_track_with_best(self):
        m = FaceTrackerManager(track_timeout=2.0)
        a = m.update([(0, 0, 10, 10)], now=0.0)
        tid = a[0][0]
        m.record_quality(tid, 1.0, np.zeros((5, 5, 3), np.uint8), (0, 0, 10, 10), 0.0)
        self.assertEqual(m.collect_finished(now=1.0), [])     # henuz timeout degil
        finished = m.collect_finished(now=10.0)               # timeout asildi
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0].track_id, tid)


if __name__ == "__main__":
    unittest.main()
