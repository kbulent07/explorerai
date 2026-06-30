from tracking import associate_faces_to_persons


def _face(x, y, w, h):
    return {"bbox": (x, y, w, h), "confidence": 0.9}


def test_face_inside_single_person():
    faces = [_face(50, 30, 20, 20)]        # merkez (60,40)
    persons = [(7, (40, 20, 60, 120))]     # (40..100, 20..140) icerir
    out = associate_faces_to_persons(faces, persons)
    assert out == [(7, faces[0])]


def test_face_without_container_is_dropped():
    faces = [_face(500, 500, 20, 20)]
    persons = [(1, (0, 0, 100, 100))]
    assert associate_faces_to_persons(faces, persons) == []


def test_multi_candidate_picks_highest_iou():
    # Yuz merkezi iki kisi kutusunda da; daha cok ortusen kazanir
    face = _face(45, 45, 20, 20)           # merkez (55,55)
    p_loose = (2, (0, 0, 120, 120))        # yuzle az ortusur (buyuk kutu)
    p_tight = (1, (40, 40, 40, 40))        # yuzu tamamen icerir -> daha yuksek IoU
    out = associate_faces_to_persons([face], [p_loose, p_tight])
    assert out == [(1, face)]


def test_tie_break_smallest_track_id():
    # Iki ozdes kisi kutusu (esit IoU) -> en kucuk track_id secilir
    face = _face(45, 45, 20, 20)
    out = associate_faces_to_persons([face], [(5, (0, 0, 120, 120)), (3, (0, 0, 120, 120))])
    assert out == [(3, face)]
