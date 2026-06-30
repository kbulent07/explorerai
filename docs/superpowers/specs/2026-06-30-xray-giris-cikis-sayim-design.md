# Tasarım: X-ray Noktası Giriş/Çıkış Kişi Sayımı (CPU-hafif)

**Tarih:** 2026-06-30
**Durum:** Onaylandı (uygulama planına hazır)
**Kapsam:** FaceZoom'a, X-ray/turnike koridorundan geçen kişileri **yön-bazlı** (giriş/çıkış) sayan, mevcut yüz/zoom hattını bozmayan, kapatılabilir bir modül eklemek.

**Güncelleme (2026-06-30, kod uyumu):** Önceki ara çözüm olan **kamera-rolü tabanlı** (per-kamera `giris`/`cikis` + yüz tanıma) sayım kodu `recent.py`/`webui.py`'den zaten çıkarılmış; `config_store`'da artık tüketicisiz kalan rol altyapısı (`role`/`camera_roles`/`_norm_role` + ayarlar ekranı rol seçici) bu modül kapsamında **temizlenecek** (yön artık çizgiden belirlenir, kameraya rol atanmaz). YOLOX modeli repoya gömülmez; bir **indirme script'i/komutu** + README notu ile sağlanır ve `.onnx` `.gitignore`'a eklenir.

---

## 1. Amaç ve kısıtlar

**Amaç:** Tek bir güvenlik kamerasının gördüğü X-ray koridorundan **fabrikaya giren** ve **fabrikadan çıkan** kişileri saymak; aynı kişiyi tekrar saymamak; mümkünse **isim bazında** giren/çıkan listesi vermek.

**Kısıtlar (kabul edilmiş kararlar):**
- **Tek kamera, tek koridor.** İki yön de aynı yerden geçer; yön, sanal çizginin hangi tarafına geçildiğinden belirlenir.
- **Hibrit:** Gövde (kişi) algılama + takip sayımın omurgası; yüz tanıma kimlik/isim içindir.
- **CPU öncelikli.** GPU sonradan, **kod değişmeden** (onnxruntime provider) açılabilir.
- **Sayım RAM'de** tutulur (kümülatif). Uygulama yeniden başlayınca sıfırlanır. Manuel "Sıfırla" düğmesi olur. Diske/DB'ye yazılmaz (mevcut gizlilik tasarımıyla tutarlı).
- **İzole + opsiyonel.** Mevcut yüz algılama / zoom / önizleme / galeri hattına dokunulmaz; `counting_enabled` ile açılır/kapanır.

**Lisans kısıtı:** Hiç AGPL bağımlılığı olmayacak.
- Detektör: **YOLOX-Nano (Apache-2.0)** ONNX. (Alternatif: NanoDet — Apache.)
- Takip/çizgi: **roboflow/supervision (MIT)** → `ByteTrack` + `LineZone`.
- Çalıştırma: **onnxruntime** (projede zaten var). `torch` gelmez.
- YOLOX ağırlığı repoya **gömülmez**; `person_model` config'te bir ONNX dosya yolu olur (kullanıcı sağlar).

---

## 2. Yüksek seviye mimari

```
counting kamerası (sub-stream, düşük çözünürlük)
  └─ CountingWorker (ayrı thread, ~3–5 Hz algılama)
        ├─ PersonDetector(YOLOX-ONNX, provider=CPU|CUDA)      [detector.py]
        ├─ ByteTrack (supervision) → kararlı track_id           [counting.py]
        ├─ LineZone.trigger(detections) → in_count / out_count  [counting.py]
        ├─ çizgi geçilince: track'in en iyi yüz kırpıntısı → ArcFace → isim  [recognition.py + recent.py]
        └─ CountingStore.record(isim, yön, zaman, küçük_resim)  (RAM)   [counting.py]
  └─ /counts.json → Sayım paneli (giriş/çıkış/içeride + isimli liste + Sıfırla)   [webui.py]
```

Mevcut [camera.py](../../../camera.py), [recognition.py](../../../recognition.py), [recent.py](../../../recent.py) yeniden kullanılır. Yeni kod izole modüllerdedir.

---

## 3. Bileşenler (her biri tek sorumluluk, ayrı test edilebilir)

### 3.1 `detector.py` → `PersonDetector`
- **Ne yapar:** Bir BGR kareden **kişi** kutuları döndürür: `[(x,y,w,h,conf), ...]` (COCO sınıf 0).
- **Nasıl:** YOLOX ONNX modelini onnxruntime ile yükler. Provider config'ten gelir (`CPUExecutionProvider` varsayılan; `CUDAExecutionProvider` mümkünse). Giriş karesi `person_input` (varsayılan 416) boyutuna letterbox ile ölçeklenir; çıktı NMS + sınıf filtresi (yalnız kişi) ile süzülür.
- **Bağımlılık:** onnxruntime, numpy, opencv. Model yoksa açık hata fırlatır (worker bunu yakalayıp sayımı devre dışı bırakır).
- **CPU notu:** INT8 quantize model desteklenir (config ile aynı yol; quantize edilmiş dosya verilebilir).

### 3.2 `counting.py` → `LineCounter`
- **Ne yapar:** Kişi kutularını alır → `ByteTrack` ile `tracker_id` atar → `LineZone.trigger()` ile çizgi geçişlerini hesaplar → `(track_id, "giris"|"cikis")` olayları üretir.
- **supervision API:** `LineZone(start, end)` çizgiyi tanımlar; `in_count` / `out_count`. Hangi yönün "giriş" sayılacağı **çizginin uç-nokta sırasıyla** (start→end vektörünün normali) belirlenir — `above/below` ile değil. Kullanıcı doğru yönü tutturamazsa `counting_swap` ile in/out etiketleri yer değiştirir. `ByteTrack(frame_rate=<gerçek işleme FPS>, track_activation_threshold=person_conf)`.
- **Bağımlılık:** supervision.

### 3.3 `counting.py` → `CountingWorker`
- **Ne yapar:** Sayım kamerasının **sub-stream**'ini okur ([camera.py](../../../camera.py)), `counting_detect_every` karede bir `PersonDetector`+`LineCounter` çalıştırır (ara kareleri ByteTrack interpolasyonla taşır), geçiş olaylarında **yüz füzyonu** yapar ve `CountingStore`'a yazar.
- **Yüz füzyonu:** Geçiş anındaki karede mevcut yüz algılama ([framing.py](../../../framing.py)) çalıştırılır; bir yüz kutusu, geçen track'in gövde kutusunun **içindeyse** o yüz kırpılır → ArcFace embedding ([recognition.py](../../../recognition.py)) → [recent.py](../../../recent.py)'deki kimlikle eşlenerek **isim** alınır. Yüz bulunamazsa olay isimsiz ("Kişi N") kaydedilir.
- **CPU notu:** Yüz embedding **yalnız geçişte** (kişi başına ~1 kez) çalışır, her karede değil.

### 3.4 `counting.py` → `CountingStore` (RAM)
- **Ne tutar:** `total_in`, `total_out`, `inside = max(0, in-out)` ve geçiş olayları listesi `{id, name, direction, ts, jpeg}` (küçük resim).
- **API:** `record(name, direction, ts, jpeg)`, `counts()` (panel için), `reset()`.
- **Benzersizlik:** Sayım track bazlıdır; her tamamlanan geçiş bir olaydır. Aynı kişi gerçekten iki kez geçerse iki olay olur (giriş + sonra çıkış doğru sayılır). "Aynı kişiyi tekrar sayma" gereği, **kimlik bazlı** (embedding) tekilleştirme ile panelde isimli özet olarak da sunulur.

---

## 4. Veri akışı (kare bazında)

1. CountingWorker sub-stream'den son kareyi alır.
2. `frame_count % counting_detect_every == 0` ise: `PersonDetector.detect(frame)` → kişi kutuları.
3. Kutular `sv.Detections`'a sarılır → `ByteTrack.update_with_detections()` → `tracker_id`.
4. `LineZone.trigger(detections)` → bu karede çizgiyi geçen track'ler + yön (`in_count`/`out_count` artar).
5. Geçen her track için: o anki kareden yüz füzyonu → isim → `CountingStore.record(...)`.
6. Panel `/counts.json`'ı 1–2 sn'de bir yoklar.

---

## 5. Yapılandırma (config.yaml — yeni `counting` bloğu)

```yaml
counting_enabled: false          # ana açma/kapama
counting_camera: ""              # sayım yapılacak kameranın adı (config'teki bir kamera)
counting_line: [0, 360, 1280, 360]   # [x1,y1,x2,y2] sub-stream koordinatlarında sanal çizgi (start->end)
counting_swap: false             # giriş/çıkış yönü ters çıkarsa true yap (UI'dan ayarlanır)
person_model: "models/yolox_nano.onnx"   # ONNX kişi-algılama modeli (kullanıcı koyar)
person_input: 416                # algılama giriş boyutu (320 = daha hızlı, 416 = daha doğru)
person_conf: 0.30                # algılama güven eşiği
counting_detect_every: 4         # kaç karede bir algılama (CPU); arası ByteTrack interpolasyonu
onnx_provider: "CPU"             # CPU | CUDA (GPU varsa)
```

`config_store` zaten yorumları koruyarak yazar; bu anahtarlar ayar bloğunda tutulur.

**Model temini:** `yolox_nano.onnx` repoya **gömülmez**. Bir indirme script'i/komutu (`models/download_yolox.py` veya README'de tek satırlık komut — BlazeFace modelleriyle aynı yaklaşım) ile sağlanır; `*.onnx` `.gitignore`'a eklenir. Model yoksa sayım devre dışı kalır (bkz. §7).

---

## 6. Arayüz (webui.py)

- **Sayım paneli (`/sayim`):** Üç sayaç — **İçeri giren / Dışarı çıkan / İçeride** — + **isim bazında giren** ve **isim bazında çıkan** listeleri (isim, saat, küçük resim) + **Sıfırla** düğmesi. `/counts.json`'ı yoklar.
- **Çizgi çizme:** Sayım kamerasının anlık görüntüsü üzerinde iki nokta tıklanarak sanal çizgi çizilir; giriş/çıkış yönü ters çıkarsa **"Yönü ters çevir"** seçeneği. `counting_line` ve `counting_swap` config'e yazılır.
- **Header linki:** İzleme/Ayarlar sayfalarına "Giriş/Çıkış Sayım" linki.
- `/counts.json` ve `/sayim` mevcut `require_auth` ile korunur.

---

## 7. Hata yönetimi

- **Model dosyası yok / yüklenemedi:** Sayım devre dışı; **uygulamanın geri kalanı çalışır**; log + panelde uyarı.
- **`onnx_provider: CUDA` ama GPU yok:** CPU'ya düş + uyarı.
- **`counting_camera` boş/geçersiz:** Sayım başlamaz; ayarlar ekranında uyarı.
- **Akış kopması:** [camera.py](../../../camera.py) yeniden bağlanır; worker beklemede kalır.
- **Yüz bulunamadı (geçişte):** Olay isimsiz kaydedilir (sayım yine doğru).

---

## 8. Test planı

**Birim:**
- Çizgi-geçiş yönü: sentetik track yörüngeleri çizgiyi iki yönde geçirilir → doğru `in`/`out`.
- `PersonDetector`: sahte ONNX çıktısıyla NMS + kişi-sınıfı filtresi.
- Yüz↔gövde eşleme: yüz kutusu gövde kutusunun içinde/dışında senaryoları.
- `CountingStore`: record/counts/reset; `inside` hesabı.

**Entegrasyon:**
- Örnek koridor videosu/RTSP ile uçtan uca: bilinen sayıda giriş/çıkış → panel doğru sayar.
- CPU ölçümü: tek kamerada `counting_detect_every` ve `person_input` ile FPS/CPU profili.

---

## 9. CPU bütçesi (tahmin)

- YOLOX-Nano (1.08 GFLOPs) @416 INT8, CPU'da ~20–30 ms/kare.
- `counting_detect_every=4` + ~12–15 fps kaynak → ~3–4 Hz algılama → ~%10'u bir çekirdeğin.
- ByteTrack ihmal edilebilir (numpy); yüz embedding seyrek (geçiş başına).
- **Sonuç:** pratikte **bir çekirdeğin altında**; mevcut çok-kameralı hatla birlikte çalışabilir. Kalabalıkta `counting_detect_every` düşürülür veya GPU açılır.

---

## 10. Sonuçlar ve sonra gözden geçirilecekler

- **Kolaylaşan:** doğru yön-bazlı sayım; isimli giriş/çıkış; mevcut hat etkilenmez; GPU'ya tek-config geçiş.
- **Zorlaşan:** yeni `supervision` bağımlılığı; ONNX model dosyası yönetimi; çizgi kalibrasyonu.
- **Sonra:** kalabalıkta doğruluk iyileştirme; istenirse RAM yerine kalıcı/tarihli rapor (KVKK kararı); birden çok sayım kamerası.

---

## 11. Uygulama adımları (writing-plans için özet)

0. **Temizlik (artık-kalan rol altyapısı):** `config_store`'dan `role`/`camera_roles()`/`_norm_role()` ve `add_camera`/`update_camera`/`add_hik_camera` `role` parametrelerini, ayarlar ekranındaki rol `<select>`'ini, `config.example.yaml`'daki `role` izlerini kaldır. (Yeni model kameraya rol atamaz.)
1. `requirements.txt` + `Dockerfile`: `supervision` bağımlılığı. `models/yolox_nano.onnx` için **indirme script'i/komutu** + README notu; `*.onnx` → `.gitignore`.
2. `detector.py` — `PersonDetector` (YOLOX-ONNX, provider seçilebilir, NMS + kişi filtresi).
3. `counting.py` — `LineCounter` (ByteTrack + LineZone), `CountingWorker`, `CountingStore`.
4. Yüz↔gövde füzyonu (geçişte ArcFace; [recent.py](../../../recent.py) ile isim).
5. `config.yaml` + `config_store`: `counting` anahtarları; çizgi + `counting_swap` yazma.
6. `webui.py`: `/counts.json`, `/sayim` paneli, çizgi-çizme UI, header linkleri, Sıfırla.
7. Birim + entegrasyon testleri.
8. CPU profili + `counting_detect_every`/`person_input` varsayılan ayarı.
