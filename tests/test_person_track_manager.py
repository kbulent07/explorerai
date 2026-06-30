# PersonTrackManager testleri. supervision kuruluysa calisir, degilse atlanir.
import pytest

sv = pytest.importorskip("supervision")

import numpy as np
from tracking import PersonTrackManager


def _det(x, y, w, h, conf=0.9):
    return {"bbox": (x, y, w, h), "confidence": conf}


def test_stable_track_id_across_frames():
    m = PersonTrackManager(track_activation_threshold=0.1, frame_rate=10,
                           track_timeout=5.0)
    # Ayni kisi hafifce hareket eder -> ID degismemeli
    ids = []
    for dx in range(0, 30, 3):
        out = m.update([_det(100 + dx, 100, 50, 120)], now=float(dx))
        if out:
            ids.append(out[0][0])
    assert len(ids) >= 3, "track yeterince aktive olmadi"
    assert len(set(ids)) == 1, f"track_id kararsiz: {ids}"


def test_empty_update_no_crash():
    m = PersonTrackManager()
    assert m.update([], now=0.0) == []


def test_collect_finished_after_timeout():
    m = PersonTrackManager(track_activation_threshold=0.1, frame_rate=10,
                           track_timeout=1.0)
    out = m.update([_det(100, 100, 50, 120)], now=0.0)
    assert out, "track olusmali"
    tid = out[0][0]
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    m.record_quality(tid, 0.5, crop, (100, 100, 50, 120), now=0.0)
    finished = m.collect_finished(now=5.0)  # timeout asildi
    assert any(tr.track_id == tid for tr in finished)
