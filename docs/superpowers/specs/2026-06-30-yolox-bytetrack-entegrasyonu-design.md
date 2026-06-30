# YOLOX Kişi Tespiti + supervision ByteTrack Entegrasyonu — Tasarım

- **Tarih:** 2026-06-30
- **Durum:** Onaylandı (uygulama planı bekliyor)
- **Kapsam:** FaceZoom tespit/takip hattına, config ile seçilebilir bir "kişi tespiti + ByteTrack" arka ucu eklemek.

---

## 1. Amaç ve gerekçe

Mevcut hat, kareler arası yüz takibini özel bir IoU + merkez-mesafe takipçisiyle (`tracking.py` → `FaceTrackerManager`) yapar. Bu yaklaşım, yüz kameraya tam dönük olmadığında, kişi uzaklaştığında veya kısa süreli örtüşmelerde track kimliğini kolayca kaybeder; bu da bir "görünüm" için birden çok kayıt veya ID atlamalarına yol açar.

Bu tasarım, tespiti **kişi (gövde)** seviyesine taşıyıp takibi **supervision ByteTrack**'e devrederek track kimliğini daha kararlı kılar. Yüz, baş dönükken/uzaktayken görünmese bile kişi track'i sürer; yüz tekrar göründüğünde best-shot o kararlı kişiye bağlı olarak güncellenir.

**Geri uyum birinci önceliktir:** Yeni davranış yalnızca config ile açıkça seçildiğinde devreye girer. Varsayılan mevcut MediaPipe hattıdır.

### Hedefler
- Örtüşme / baş dönüşü / uzaklık altında kararlı kişi `track_id`.
- Mevcut best-shot ve kimlik (ArcFace) mantığını koruyarak en az yeniden yazım.
- CPU-önce, GPU'ya (onnxruntime provider) hazır; torch bağımlılığı **yok**.
- A/B karşılaştırma: eski ve yeni arka uç config ile değiştirilebilir.

### Hedef olmayanlar (YAGNI)
- Yüz tanıma/kimlik mantığının değiştirilmesi (ArcFace + `RecentFaceStore` aynen kalır).
- GPU'nun bu iş kapsamında kurulması/zorunlu kılınması (yalnızca config ile hazır bırakılır).
- Çok-nesneli (kişi dışı sınıf) tespit; yalnızca COCO `person` (class 0) kullanılır.

---

## 2. Rol paylaşımı (değişmeyen ilkeler)

| Katman | Sorumluluk | Durum |
| --- | --- | --- |
| **YOLOX** | Kişi (gövde) tespiti, detect-görüntü koordinatında bbox | Yeni |
| **ByteTrack** | Kamera-içi, görünüm-başına kararlı `track_id` | `FaceTrackerManager`'ın yerini alır |
| **MediaPipe yüz** | Kişi kutusu içinde yüz bbox + landmark (frontallik için) | Korunur |
| **Best-shot (`compute_quality`)** | En net yüz karesi seçimi; artık kişi `track_id` başına | Mantık aynen |
| **ArcFace / `RecentFaceStore`** | Kameralar/görünümler arası kimlik birleştirme | **Değişmez** |

İki kimlik kavramı net ayrılır:
- **ByteTrack `track_id`** = kısa vadeli, kamera-içi "tek sürekli görünüm" → "görünüm başına tek best-shot" mantığını sürer.
- **ArcFace embedding** (`RecentFaceStore`) = uzun vadeli, kameralar arası "aynı kişi mi" → değişmez.

---

## 3. Mimari ve bileşenler

### 3.1 `detection_backend.py` (yeni dosya)

Tespit arka uçları için ince bir soyutlama ve YOLOX uygulaması.

**Arayüz:**
```
class PersonDetector:
    def detect(self, frame_bgr) -> list[dict]:
        # dönüş: [{"bbox": (x, y, w, h), "confidence": float}, ...]
        # koordinatlar detect-görüntü (det_img) uzayında
```

**`YoloxPersonDetector(PersonDetector)`:**
- ONNX modelini **tembel** yükler (`recognition.py`'deki `_ensure` deseni gibi): model dosyası yoksa veya yüklenemezse uygulama yine açılır; net hata loglanır ve `worker.py` MediaPipe arka ucuna düşer.
- onnxruntime `InferenceSession`, provider listesi config'ten (`yolox_providers`). CUDA provider mevcut değilse onnxruntime CPU'ya düşer; bu durum loglanır.
- **Ön-işleme:** letterbox resize → `(input_size, input_size)`, en-boy oranı korunur, padding değeri `114`, normalizasyon **yok** (standart YOLOX ONNX export sözleşmesi). Ölçek faktörü `r` saklanır.
- **Son-işleme:** grid decode (strides 8/16/32) → `(cx, cy, w, h, obj, 80 class)` → skor = `obj * class`, `yolox_confidence` eşiği, **class 0 (person) filtresi**, numpy NMS (`yolox_nms`). Kutular `r` ile detect-görüntü uzayına geri ölçeklenir ve `(x, y, w, h)` formatına çevrilir.
- `person_min_size` altındaki kişiler elenir.

> Not: Decode/NMS, resmi YOLOX `demo/ONNXRuntime/onnx_inference.py` mantığını izler ve **saf numpy** ile, ağır bağımlılık olmadan yazılır → birim testi sentetik tensörle yapılabilir.

### 3.2 `tracking.py` (genişletme)

- Yeni `PersonTrackManager`:
  - İçeride `supervision.ByteTrack` örneği tutar; `update(person_boxes, now)` çağrısında person kutularını `sv.Detections`'a çevirir, `update_with_detections()` ile `track_id` alır ve `worker.py`'nin beklediği `[(track_id, bbox), ...]` listesini döndürür.
  - Mevcut `Track` sınıfını ve best-shot makinesini (`maybe_update_best`, `record_quality`, `collect_finished`, `flush_all`) **yeniden kullanır**. Public yüzey `FaceTrackerManager` ile **birebir aynıdır** → `worker.py` değişimi minimum.
  - **Görünüm sonlandırma:** mevcut `track_timeout` / `last_seen` mantığı korunur (ByteTrack'in iç track yaşam döngüsünden bağımsız). Böylece "biten görünümü finalize et → best-shot'ı yayınla" davranışı bugünküyle tutarlı kalır.
- Yeni saf fonksiyon `associate_faces_to_persons(faces, person_tracks)`:
  - Her yüz, **merkezi içinde bulunan** kişi kutusuna eşlenir; birden çok aday varsa en yüksek IoU'lu kişi seçilir (belirleyici: eşitlikte en küçük `track_id`).
  - Dönüş: `[(track_id, face), ...]`. İçeren kişisi olmayan yüzler dışarıda bırakılır (sayısı loglanır).
  - Saf/yan-etkisiz → hızlı birim testi.

### 3.3 `worker.py` (değiştirme)

Başlangıçta `detector_backend` config değerine göre `self.manager` ve tespit yolu seçilir:

- **`mediapipe` (varsayılan):** Bugünkü davranış **aynen**. `FaceTrackerManager` + MediaPipe yüzleri doğrudan detect kutuları olarak kullanılır. Hiçbir değişiklik yoktur.
- **`yolox_person`:** detect bloğunda (her `detect_interval` karede bir):
  1. `YoloxPersonDetector.detect(det_img)` → kişi kutuları.
  2. `PersonTrackManager.update(person_boxes, now)` → `[(track_id, person_bbox)]`.
  3. MediaPipe yüz tespiti `det_img` üzerinde (bugünkü gibi tüm karede).
  4. `associate_faces_to_persons(faces, person_tracks)` → her yüz kişi `track_id`'sini devralır.
  5. Eşleşen her (track_id, face) için: hires kırp (`crop_with_margin`), `compute_quality`, `record_quality(track_id, ...)`. (Mevcut best-shot akışı, yalnız anahtar artık kişi track_id.)
  6. Yüzü görünmeyen kişiler: track sürer, best-shot güncellenmez.
- `collect_finished` / `_emit_capture` her iki yolda da **aynı şekil**.
- **Canlı zoom hedefi:** en büyük **yüz** (mevcut davranış korunur — varsayılan karar). Alternatif "en büyük kişi" açık konu olarak (§8) bırakılır.

### 3.4 `recognition.py` / `recent.py`

**Değişmez.** Best-shot yüz kırpıntısı yine `RecognitionPipeline` → ArcFace → `RecentFaceStore` yoluna girer.

---

## 4. Veri akışı (yolox_person modu)

```
RTSP detect akışı
   └─> det_img (downscale)
         ├─> YOLOX ────────────> kişi kutuları ──> ByteTrack ──> (track_id, person_bbox)
         └─> MediaPipe yüz ────> yüz kutuları ───┐
                                                 v
                            associate_faces_to_persons (içerme + IoU)
                                                 v
                                  (track_id, face) eşleşmeleri
                                                 v
                  hires kırp ──> compute_quality ──> record_quality(track_id)
                                                 v
                       collect_finished (track_timeout) ──> _emit_capture
                                                 v
                          RecognitionPipeline (ArcFace) ──> RecentFaceStore
```

---

## 5. Config eklentileri

`config.yaml`, `config.example.yaml` ve `config_store.py` doğrulamasına eklenir:

```yaml
# ---- tespit arka ucu ----
detector_backend: mediapipe        # varsayılan (geri uyumlu) | yolox_person
yolox_model: "models/yolox_nano.onnx"
yolox_input_size: 416              # nano/tiny varsayılanı
yolox_confidence: 0.35             # kişi skoru eşiği
yolox_nms: 0.45                    # NMS IoU eşiği
yolox_providers: ["CPUExecutionProvider"]   # GPU: + "CUDAExecutionProvider"
person_min_size: 40                # px (detect uzayı); küçük kişileri ele
# ---- ByteTrack ----
bytetrack_track_activation_threshold: 0.25
bytetrack_lost_buffer: 30          # kaç kare kayıp track tutulsun
bytetrack_min_matching_threshold: 0.8
```

**Doğrulama (`config_store.py`):** `detector_backend` ∈ {`mediapipe`, `yolox_person`}; sayısal alanlar tip/aralık kontrolü; `yolox_providers` liste-of-string. Geçersiz değerler güvenli varsayılana düşer ve uyarı loglanır.

---

## 6. Bağımlılıklar

`requirements.txt`:
- `supervision` (ByteTrack'i getirir). Çekirdek ByteTrack numpy/scipy tabanlıdır — **torch'suz**. Kurulumda kesin transitive bağımlılık ayak izi doğrulanacak (matplotlib gibi gereksiz ağır paket gelirse not düşülecek).
- `onnxruntime` **zaten var**; YOLOX çıkarımı elle yazılır → `yolox` pip paketi ve torch **gerekmez**.

**Model dosyası:** `yolox_nano.onnx` (varsayılan) repoya **konmaz** (`.gitignore`/dağıtım dışı). README ve DOCKER.md'ye indirme/edinme adımı eklenir. (nano, CPU için varsayılan; daha hızlı/az isabetli alternatif: tiny.)

---

## 7. Hata yönetimi (canlı, CPU-bound → asla çökmesin)

- YOLOX modeli yok/yüklenemiyor → hata loglanır, **`mediapipe` arka ucuna fallback**; webui durumunda gösterilir.
- CUDA provider yok → onnxruntime CPU'ya düşer, uyarı loglanır.
- `supervision` kurulu değilken `yolox_person` seçilmiş → log + mediapipe fallback.
- İçeren kişisi olmayan yüz → `yolox_person` modunda düşürülür (sayısı debug loglanır).
- YOLOX/MediaPipe tek karede hata verirse → o kare atlanır, dongü devam eder (canlılık önceliği; mevcut desen).

---

## 8. Açık konular (uygulamada karara bağlanacak / izlenecek)

- **Canlı zoom hedefi:** en büyük yüz (varsayılan) vs en büyük kişi. Şu an: en büyük yüz.
- **Performans bütçesi:** YOLOX-nano@416 CPU'da kare başına on-ms'ler beklenir; `detect_interval=2` + `detect_downscale=0.5` ile 1-2 kamerada `preview_fps=12` hedefine sığması beklenir. Sığmazsa: `detect_interval` ↑ ya da `yolox_input_size` ↓. Uygulamada ölçülecek; gerçek sayılar buraya eklenecek.
- **webui `/settings`:** backend seçici + YOLOX alanları. Bu spec kapsamında tasarlanır; uygulama planında ele alınır.

---

## 9. Test stratejisi

- `tests/test_yolox_postprocess.py`: sentetik YOLOX çıktı tensörü → decode + NMS + class-0 filtresi + koordinat geri-ölçekleme doğrulanır. Ağır bağımlılık import etmez.
- `tests/test_face_person_association.py`: saf `associate_faces_to_persons` — içerme, çoklu aday IoU seçimi, eşitlikte belirleyicilik, içeren-kişisiz yüzün düşmesi.
- `tests/test_person_track_manager.py`: kutu dizileriyle kararlı `track_id`, best-shot kaydı ve `collect_finished` zaman aşımı. (supervision kuruluysa gerçek ByteTrack; değilse test atlanır/guard'lanır — mevcut `test_tracking.py`'nin ağır-import-etmeme yaklaşımına uyumlu.)
- **Regresyon:** mevcut testler (`test_tracking.py`, `test_recent.py`, `test_config_store.py`) yeşil kalmalı; mediapipe yolu değişmedi.

---

## 10. Geri uyum ve aşamalı geçiş

- Varsayılan `detector_backend: mediapipe` → kullanıcı opt-in yapmadıkça **sıfır davranış değişikliği**.
- Yeni dosya/bağımlılık eski yolu etkilemez (tembel import, koşullu instantiation).
- `yolox_person` opt-in edildiğinde model yoksa otomatik fallback → kullanıcı kademeli geçiş yapabilir.

---

## 11. Etkilenen dosyalar (özet)

| Dosya | Değişiklik |
| --- | --- |
| `detection_backend.py` | **Yeni** — `PersonDetector` arayüzü + `YoloxPersonDetector` |
| `tracking.py` | **Genişletme** — `PersonTrackManager`, `associate_faces_to_persons` |
| `worker.py` | **Değiştirme** — `detector_backend`'e göre dallanma, yüz↔kişi eşleme |
| `config.yaml`, `config.example.yaml` | **Ekleme** — yeni anahtarlar |
| `config_store.py` | **Ekleme** — yeni anahtarların doğrulaması |
| `requirements.txt` | **Ekleme** — `supervision` |
| `README.md`, `DOCKER.md` | **Ekleme** — YOLOX modeli edinme adımı |
| `tests/` | **Yeni** — 3 test dosyası |
| `recognition.py`, `recent.py`, `framing.py` | **Değişmez** |
