# tests/test_counting.py
# LineCrossingCounter (yon tespiti) + CountingStore (sayac) birim testleri.

from counting import LineCrossingCounter, CountingStore


def _bbox_at(cx, cy, w=20, h=40):
    """Merkezi (cx,cy) olan (x,y,w,h) kutusu."""
    return (int(cx - w / 2), int(cy - h / 2), w, h)


# --- LineCrossingCounter ---------------------------------------------------

# Normalize yatay cizgi y=0.5; dims=(200,200) -> piksel y=100
_LINE = (0.0, 0.5, 1.0, 0.5)
_DIMS = (200, 200)


def test_yatay_cizgi_giris_ve_cikis():
    counter = LineCrossingCounter(line=_LINE)
    # Kisi ustten alta iner (y: 50 -> 150) => bir yon
    assert counter.update([(1, _bbox_at(100, 50))], _DIMS) == []   # ilk kare: taraf belirlenir
    ev = counter.update([(1, _bbox_at(100, 150))], _DIMS)          # cizgiyi gecti
    assert len(ev) == 1 and ev[0][0] == 1
    first_dir = ev[0][1]
    assert first_dir in ("giris", "cikis")
    # Geri yukari cikar (150 -> 50) => ters yon
    ev2 = counter.update([(1, _bbox_at(100, 50))], _DIMS)
    assert len(ev2) == 1
    assert ev2[0][1] != first_dir


def test_cizgiyi_gecmeden_olay_yok():
    counter = LineCrossingCounter(line=_LINE)
    counter.update([(1, _bbox_at(50, 30))], _DIMS)
    # Ayni tarafta kalir (y hep < 100) -> gecis yok
    assert counter.update([(1, _bbox_at(60, 40))], _DIMS) == []
    assert counter.update([(1, _bbox_at(70, 20))], _DIMS) == []


def test_swap_yonu_ters_cevirir():
    a = LineCrossingCounter(line=_LINE, swap=False)
    b = LineCrossingCounter(line=_LINE, swap=True)
    a.update([(1, _bbox_at(100, 50))], _DIMS)
    b.update([(1, _bbox_at(100, 50))], _DIMS)
    da = a.update([(1, _bbox_at(100, 150))], _DIMS)[0][1]
    db = b.update([(1, _bbox_at(100, 150))], _DIMS)[0][1]
    assert da != db


def test_farkli_tracklar_bagimsiz():
    counter = LineCrossingCounter(line=_LINE)
    counter.update([(1, _bbox_at(100, 50)), (2, _bbox_at(120, 50))], _DIMS)
    ev = counter.update([(1, _bbox_at(100, 150)), (2, _bbox_at(120, 150))], _DIMS)
    ids = sorted(e[0] for e in ev)
    assert ids == [1, 2]


def test_forget_track_durumu_temizler():
    counter = LineCrossingCounter(line=_LINE)
    counter.update([(1, _bbox_at(100, 50))], _DIMS)
    counter.forget([1])
    # Durum unutuldu -> alta inse de "onceki taraf yok" => olay yok
    assert counter.update([(1, _bbox_at(100, 150))], _DIMS) == []


# --- CountingStore ---------------------------------------------------------

def test_sayac_giris_cikis_iceride():
    s = CountingStore()
    s.record("giris", ts=1.0, name="Ahmet")
    s.record("giris", ts=2.0, name="Mehmet")
    s.record("cikis", ts=3.0, name="Ahmet")
    c = s.counts()
    assert c["in"] == 2
    assert c["out"] == 1
    assert c["inside"] == 1
    assert len(c["entered"]) == 2
    assert len(c["exited"]) == 1


def test_gecersiz_yon_sayilmaz():
    s = CountingStore()
    assert s.record("bilinmeyen") is None
    assert s.counts()["in"] == 0


def test_jpeg_saklanir_ve_alinir():
    s = CountingStore()
    eid = s.record("giris", jpeg=b"JPEGDATA")
    assert s.get_jpeg(eid) == b"JPEGDATA"
    assert s.get_jpeg(99999) is None


def test_reset_sifirlar():
    s = CountingStore()
    s.record("giris")
    s.record("cikis")
    s.reset()
    c = s.counts()
    assert c["in"] == 0 and c["out"] == 0 and c["inside"] == 0
    assert c["entered"] == [] and c["exited"] == []


def test_max_events_tasmasi():
    s = CountingStore(max_events=3)
    for i in range(5):
        s.record("giris", ts=float(i))
    # Sayac toplami artar ama yalniz son 3 olay listede tutulur
    c = s.counts()
    assert c["in"] == 5
    assert len(c["entered"]) == 3
