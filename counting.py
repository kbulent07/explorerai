# counting.py
# -----------------------------------------------------------------------------
# Cizgi-gecis tabanli GIRIS/CIKIS sayimi (yon-bazli), bellek-ici (RAM).
#
# Tek kamera, tek koridor senaryosu: sanal bir cizgi tanimlanir; takip edilen
# her kisinin (track) merkezi bu cizgiyi hangi YONDE gecerse o yon "giris" ya da
# "cikis" olarak sayilir. Sayim TRACK bazlidir -> ByteTrack kararli track_id
# verdigi surece ayni gecis tekrar sayilmaz.
#
# Backend-bagimsiz: hem PersonTrackManager (yolox_person/ByteTrack) hem
# FaceTrackerManager [(track_id,(x,y,w,h)), ...] ciktisi verdiginden ikisinde de
# calisir. Yon, cizginin uc-nokta sirasiyla belirlenir; ters cikarsa `swap`.
#
# Diske YAZILMAZ; uygulama yeniden baslayinca sayim sifirlanir (oturum-ici).
# -----------------------------------------------------------------------------

import threading
import time


def _signed_side(p1, p2, pt):
    """p1->p2 dogrusuna gore pt'nin isaretli tarafi (2B capraz carpim).
    >0 bir taraf, <0 diger taraf, 0 tam dogru uzerinde."""
    return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])


class LineCrossingCounter:
    """Track merkezlerinin sanal cizgiyi hangi yonde gectigini tespit eder.

    line : (x1, y1, x2, y2) -> cizginin iki ucu (detect/işleme koordinatlari).
    swap : giris/cikis yonu ters cikarsa True (UI'dan ayarlanir).

    update(tracks) bu karede gecis yapan track'ler icin
    [(track_id, "giris"|"cikis"), ...] dondurur. Her track icin son tarafi
    hatirlar; isaret degisince (cizgiyi gecince) bir olay uretir.
    """

    def __init__(self, line, swap=False, forget_after=60.0):
        # line NORMALIZE koordinattir (0..1): [x1,y1,x2,y2]. Boylece detect
        # cozunurlugu ne olursa olsun ayni cizgi gecerli (kameraya tikla-ciz UI
        # de normalize uretir). Piksele cevrim update() icinde dims ile yapilir.
        x1, y1, x2, y2 = line
        self.n1 = (float(x1), float(y1))
        self.n2 = (float(x2), float(y2))
        self.swap = bool(swap)
        # track_id -> [son_isaret(+1/-1), son_gorulme_ts]. ByteTrack id'leri MONOTON
        # artar; bu sozluk update()'te otomatik budanmazsa uzun sureli sayimda
        # SINIRSIZ buyur (bellek sizintisi). forget_after sn goruunmeyen track silinir.
        self._last_sign = {}
        self._forget_after = float(forget_after)

    def update(self, tracks, dims):
        """tracks: [(track_id, (x,y,w,h)), ...] (detect piksel).
        dims: (w, h) detect uzayi boyutu -> normalize cizgi piksele cevrilir.
        Gecis olaylarini [(track_id, 'giris'|'cikis'), ...] dondurur."""
        now = time.time()
        w_d, h_d = float(dims[0]), float(dims[1])
        p1 = (self.n1[0] * w_d, self.n1[1] * h_d)
        p2 = (self.n2[0] * w_d, self.n2[1] * h_d)
        events = []
        for tid, bbox in tracks:
            x, y, w, h = bbox
            cx = x + w / 2.0
            cy = y + h / 2.0
            s = _signed_side(p1, p2, (cx, cy))
            sign = 1 if s > 0 else (-1 if s < 0 else 0)
            if sign == 0:
                continue  # tam cizgi uzerinde: kararsiz, atla
            prev = self._last_sign.get(tid)
            self._last_sign[tid] = [sign, now]
            if prev is not None and prev[0] != sign:
                # -1 -> +1 yonu "giris" kabul; swap ile ters cevrilir
                direction = "giris" if sign > 0 else "cikis"
                if self.swap:
                    direction = "cikis" if direction == "giris" else "giris"
                events.append((tid, direction))
        # Uzun suredir gorulmeyen track durumlarini unut (bellek sismesin).
        cutoff = now - self._forget_after
        stale = [t for t, v in self._last_sign.items() if v[1] < cutoff]
        for t in stale:
            del self._last_sign[t]
        return events

    def forget(self, track_ids):
        """Biten track'lerin durumunu unut (bellek sismesin)."""
        for tid in track_ids:
            self._last_sign.pop(tid, None)


class CountingStore:
    """Giris/cikis sayaclari + son gecis olaylari (RAM, thread-guvenli).

    Olaylar yeni ustte tutulur; en fazla `max_events` saklanir. Her olay
    opsiyonel bir kucuk resim (jpeg) ve isim tasiyabilir.
    """

    def __init__(self, max_events=500):
        self._lock = threading.Lock()
        self._max = int(max_events)
        self.total_in = 0
        self.total_out = 0
        self._events = []   # [{id, direction, ts, name, camera, jpeg}]
        self._eid = 1
        # Teshis: sayacin ne gordugu (track var mi, gecis kontrol ediliyor mu)
        self.track_obs = 0       # track goruldugu kare sayisi (>0 track)
        self.last_tracks = 0     # son karede track sayisi
        self.crossings = 0       # toplam gecis olayi

    def note(self, n_tracks):
        """Her algilama karesinde track sayisini kaydet (teshis icin)."""
        with self._lock:
            self.last_tracks = int(n_tracks)
            if n_tracks > 0:
                self.track_obs += 1

    def record(self, direction, ts=None, name=None, camera=None, jpeg=None):
        """Bir gecis olayini kaydet. direction: 'giris'|'cikis'. Olay id'si doner."""
        if direction not in ("giris", "cikis"):
            return None
        ts = time.time() if ts is None else float(ts)
        with self._lock:
            if direction == "giris":
                self.total_in += 1
            else:
                self.total_out += 1
            self.crossings += 1
            eid = self._eid
            self._eid += 1
            self._events.insert(0, {
                "id": eid, "direction": direction, "ts": ts,
                "name": name, "camera": camera, "jpeg": jpeg,
            })
            del self._events[self._max:]   # tasarsa en eskileri at
            return eid

    def counts(self):
        """Panel icin ozet: sayaclar + giren/cikan olay listeleri (resimsiz)."""
        with self._lock:
            def row(e):
                return {"id": e["id"], "direction": e["direction"], "ts": e["ts"],
                        "name": e["name"], "camera": e["camera"]}
            entered = [row(e) for e in self._events if e["direction"] == "giris"]
            exited = [row(e) for e in self._events if e["direction"] == "cikis"]
            return {
                "in": self.total_in,
                "out": self.total_out,
                "inside": max(0, self.total_in - self.total_out),
                "entered": entered,
                "exited": exited,
                "diag": {"track_obs": self.track_obs, "last_tracks": self.last_tracks,
                         "crossings": self.crossings},
            }

    def set_name(self, eid, name):
        """Bir olayin ismini SONRADAN guncelle (asenkron isim cozumu icin).
        Isim cozumu (ArcFace embed) ana donguyu bloke etmesin diye olay once
        isimsiz kaydedilir; cozulunce bu metodla adi yazilir. Bulundu mu doner."""
        name = (name or "").strip() or None
        if name is None:
            return False
        with self._lock:
            for e in self._events:
                if e["id"] == int(eid):
                    e["name"] = name
                    return True
        return False

    def get_jpeg(self, eid):
        with self._lock:
            for e in self._events:
                if e["id"] == int(eid):
                    return e["jpeg"]
        return None

    def reset(self):
        """Sayaclari ve olaylari sifirla (paneldeki 'Sifirla' dugmesi)."""
        with self._lock:
            self.total_in = 0
            self.total_out = 0
            self._events.clear()
