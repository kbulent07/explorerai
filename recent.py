# recent.py
# -----------------------------------------------------------------------------
# Bellek-ici "son yuzler" deposu (diske YAZMAZ). KAMERA BASINA ayri tutulur.
#
# Yuz TANIMA yoktur. "Ayni kisi" yaklasik olarak KONUM yakinligi ile belirlenir:
# masada sabit oturan birinin yuz merkezi kareler arasi yaklasik aynidir.
#
# Davranis:
#   - Her kamera icin en cok `per_camera_limit` (varsayilan 50) kisi tutulur.
#     Sinir assilinca en eski (en uzun suredir gorulmeyen) kisi dussar.
#     ZAMAN ASIMI YOKTUR; kisi 2 dk guncellenmese de listede kalir.
#   - Gosterilecek kare, kisinin SON `best_window` saniye (varsayilan 120 = 2 dk)
#     icindeki EN NET karesidir. Bu sure yalniz "hangi kare" secimi icindir;
#     gosterimin ekranda kalma suresiyle ilgisi yoktur.
#
# Goruntuler JPEG olarak RAM'de tutulur (kisi basina tek "en iyi" kare).
# -----------------------------------------------------------------------------

import threading
import time

import numpy as np


class RecentFaceStore:
    """Kisi = bir KIMLIK. Eslessme oncelikle YUZ EMBEDDING'i (kosinus benzerligi)
    ile yapilir -> ayni kisi farkli konumda/kamerada bile TEK kayit olur (yuz
    tanima). Embedding yoksa (tanima basarisiz) konum yakinligina dussulur.

    Embedding'ler yalniz BELLEKTE tutulur; diske yazilmaz.
    """

    def __init__(self, best_window=120, max_bytes=256 * 1024 * 1024,
                 match_dist_factor=1.3, sim_threshold=0.40):
        self.best_window = best_window            # en-net kare secimi penceresi (sn)
        self.max_bytes = int(max_bytes)           # bellek butcesi (RAM'e gore belirlenir)
        self.match_factor = match_dist_factor     # konum yedek eslessme
        self.sim_threshold = sim_threshold        # kosinus benzerlik esigi (tanima)
        self._lock = threading.Lock()
        self._entries = {}      # id -> entry dict
        self._next_id = 1
        self._bytes = 0         # tutulan JPEG'lerin toplam boyutu
        self._person_seq = 1    # otomatik isim sayaci ("Kisi N")

    @staticmethod
    def _centroid(bbox):
        x, y, w, h = bbox
        return x + w / 2.0, y + h / 2.0, w, h

    def add(self, camera, bbox, jpeg, quality, ts=None, embedding=None):
        """Yeni bir en-net kare ekle. (entry_id, yeni_kisi_mi) dondurur.

        embedding: L2-normalize edilmiss yuz vektoru (np.ndarray) veya None.
        """
        ts = time.time() if ts is None else ts
        cx, cy, w, h = self._centroid(bbox)
        with self._lock:
            match = None

            # 1) YUZ TANIMA: tum kayitlarda en yuksek kosinus benzerligini bul
            if embedding is not None:
                best_sim = self.sim_threshold
                for e in self._entries.values():
                    emb = e.get("emb")
                    if emb is None:
                        continue
                    sim = float(np.dot(emb, embedding))
                    if sim >= best_sim:
                        best_sim = sim
                        match = e

            # 2) Konum yedegi YALNIZ embedding YOKKEN (tanima kapali/basarisiz).
            #    Embedding VARSA ve adim 1'de eslesmediyse, tanima "bu farkli bir
            #    kisi" demis demektir -> konuma bakip BIRLESTIRME (aksi halde ayni
            #    koltukta oturan iki farkli kisi tek kimlige cokerdi ve EMA ile
            #    embedding kirlenirdi). Bu durumda asagida YENI kimlik acilir.
            if match is None and embedding is None:
                for e in self._entries.values():
                    if e["camera"] != camera:
                        continue
                    dist = ((cx - e["cx"]) ** 2 + (cy - e["cy"]) ** 2) ** 0.5
                    if dist <= self.match_factor * max(w, e["w"], 1):
                        match = e
                        break

            if match is None:
                eid = self._next_id
                self._next_id += 1
                auto_name = f"Kisi {self._person_seq}"   # otomatik isim (degisstirilebilir)
                self._person_seq += 1
                self._entries[eid] = {
                    "id": eid, "camera": camera,
                    "cx": cx, "cy": cy, "w": w, "h": h,
                    "emb": embedding,
                    "jpeg": jpeg, "quality": float(quality), "bytes": len(jpeg),
                    "best_ts": ts, "first_seen": ts, "last_seen": ts,
                    "name": auto_name,   # oturum-ici (RAM) isim; embedding eslessince korunur
                }
                self._bytes += len(jpeg)
                self._evict_to_budget()
                return eid, True

            # Mevcut kisi: konum/kamera tazele
            match["last_seen"] = ts
            match["camera"] = camera
            match["cx"], match["cy"], match["w"], match["h"] = cx, cy, w, h
            # Embedding'i yumuscak guncelle (EMA) -> kimlik kararliligi
            if embedding is not None:
                if match.get("emb") is None:
                    match["emb"] = embedding
                else:
                    blended = 0.8 * match["emb"] + 0.2 * embedding
                    n = np.linalg.norm(blended)
                    match["emb"] = blended / n if n > 0 else embedding
            # SON 2 dk'nin en net karesi: daha kaliteliyse VEYA mevcut en iyi
            # kare pencereden cikacak kadar eskidiyse degissir.
            if quality > match["quality"] or (ts - match["best_ts"]) > self.best_window:
                self._bytes += len(jpeg) - match.get("bytes", 0)
                match["jpeg"] = jpeg
                match["bytes"] = len(jpeg)
                match["quality"] = float(quality)
                match["best_ts"] = ts
            return match["id"], False

    def _evict_to_budget(self):
        """Bellek butcesi assilirsa en uzun suredir gorulmeyen kayitlari dussur."""
        if self._bytes <= self.max_bytes:
            return
        # en eski (en kucuk last_seen) once
        order = sorted(self._entries.values(), key=lambda e: e["last_seen"])
        i = 0
        while self._bytes > self.max_bytes and len(self._entries) > 1 and i < len(order):
            e = order[i]; i += 1
            self._bytes -= e.get("bytes", 0)
            del self._entries[e["id"]]

    def drop_camera(self, camera):
        """Bir kameraya ait TUM kayitlari sil (kamera config'ten silinince).

        Aksi halde silinen kameranin eski en-net kareleri bellekte kalip galeri/
        popup'ta gorunmeye devam eder. Silinen kayit sayisini dondurur.
        """
        with self._lock:
            ids = [eid for eid, e in self._entries.items() if e["camera"] == camera]
            for eid in ids:
                self._bytes -= self._entries[eid].get("bytes", 0)
                del self._entries[eid]
            return len(ids)

    def stats(self):
        with self._lock:
            return {"count": len(self._entries),
                    "bytes": self._bytes,
                    "max_bytes": self.max_bytes}

    def list_recent(self):
        """Tum kisiler; en son gorulen ustte. (Kamera bazinda frontend gruplar.)"""
        with self._lock:
            items = sorted(self._entries.values(),
                           key=lambda e: e["last_seen"], reverse=True)
            return [
                {
                    "id": e["id"], "camera": e["camera"],
                    "quality": round(e["quality"], 3),
                    "best_ts": e["best_ts"], "first_seen": e["first_seen"],
                    "last_seen": e["last_seen"],
                    "name": e.get("name"),
                }
                for e in items
            ]

    def set_name(self, eid, name):
        """Bir kimlige RAM'de isim ver (bossaltmak icin bos string). Bu kimlik
        embedding ile esleendigi surece kisi tekrar gelince ISIM KORUNUR.
        Diske YAZILMAZ -> yeniden baslayinca silinir (oturum-ici)."""
        name = (name or "").strip() or None
        with self._lock:
            e = self._entries.get(int(eid))
            if e is None:
                return False
            e["name"] = name
            return True

    def name_of(self, eid):
        """Bir kimligin adini dondur (yoksa None)."""
        with self._lock:
            e = self._entries.get(int(eid))
            return e.get("name") if e else None

    def name_for_embedding(self, embedding):
        """Verilen embedding'e en cok benzeyen (>= sim_threshold) ISIMLI kimligin
        adini dondur. Giris/cikis olayini bir kisiye baglamak icin kullanilir.
        Eslesme yoksa None."""
        if embedding is None:
            return None
        with self._lock:
            best = self.sim_threshold
            name = None
            for e in self._entries.values():
                emb = e.get("emb")
                if emb is None or not e.get("name"):
                    continue
                sim = float(np.dot(emb, embedding))
                if sim >= best:
                    best = sim
                    name = e["name"]
            return name

    def get_jpeg(self, eid):
        with self._lock:
            e = self._entries.get(int(eid))
            return e["jpeg"] if e else None
