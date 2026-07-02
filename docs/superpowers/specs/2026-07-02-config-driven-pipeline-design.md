# Config-Driven Plugin Pipeline — Tasarım (Aşama 1)

**Tarih:** 2026-07-02
**Kapsam:** FaceZoom'u çok-işletmeli gerçek zamanlı CV gözetim mimarisine aşamalı
uyarlamanın **1. aşaması**: kameranın işlem hattını (pipeline) YAML ile tanımlanan
bir modül zincirine çevirmek. Bu belge yalnız Aşama 1'i kapsar.

## 1. Amaç ve Bağlam

Referans mimari, her kameranın işlem hattını YAML'da bir modül listesi olarak
tanımlar ("Yeni özellik = yeni modül dosyası + config'e bir satır"). FaceZoom'da
bu mantık şu an `worker.CameraWorker.process()` içine gömülü ve sabittir:
tespit → takip → best-shot → tanıma (`_on_capture`) → sayım (`_run_counting`) →
zoom → overlay → resize.

Bu aşama, o hattı **iki katmana** ayırır:

- **Algılama çekirdeği (SABİT):** detect/hires kare okuma, koordinat ölçekleme
  (sx, sy), yüz/kişi tespiti, ByteTrack, best-shot kaydı, `collect_finished`.
  Koordinat-hassas olduğu için modülleştirilmez; `ctx` sözlüğünü doldurur.
- **Modül zinciri (CONFIG):** çekirdek `ctx`'i doldurduktan sonra `pipeline:`
  listesindeki modüller çalışır.

Bu aşama **davranış-korumalı bir refactor**tur: yeni özellik eklemez; mevcut
tanıma/sayım/zoom/overlay davranışını modüllere sarar ve `pipeline:` tanımlı
değilse bugünkü çıktıyı birebir üretir.

### Amaç dışı (bu aşamada YOK)
- Merkezi ZMQ/shared-memory GPU inference pool (Aşama 5).
- Multi-tenant / işletme yapısı (Aşama 3).
- Process-level self-healing / supervisor (Aşama 4).
- Yeni analiz/alarm özellikleri (Aşama 2: Raporlama/Alarm bir modül olarak).
- Dedektör/tracker'ın modülleştirilmesi (bilinçli olarak çekirdekte kalır).

## 2. Mimari

`CameraWorker.process()` iki faza ayrılır:

```
[ALGILAMA ÇEKİRDEĞİ - sabit]
  read_detect / read_hires
  -> koordinat uzayı + ölçek (sx, sy)
  -> tracker.detect (MediaPipe)  [+ YOLOX kişi + associate  (yolox_person)]
  -> best-shot kaydı (record_quality)
  -> collect_finished
  => ctx doldurulur

[MODÜL ZİNCİRİ - config]
  faz A (analiz):   her modul.process(ctx)   liste sırasıyla
  faz B (display):  her modul.draw(ctx)      liste sırasıyla
  => çekirdek ctx["output"]'u output_size'a resize eder ve döndürür
```

İki geçiş (önce tüm `process()`, sonra tüm `draw()`) analizin çizimden önce
tamamlanmasını garanti eder; çizim sırası liste sırasıyla belirlenir.

## 3. Modül Arayüzü (`PipelineModule`)

```python
class PipelineModule:
    def setup(self, config, camera): ...   # opsiyonel: bir kez kurulum
    def process(self, ctx): ...            # opsiyonel: ANALİZ, ctx'i günceller
    def draw(self, ctx): ...               # opsiyonel: DISPLAY, ctx["output"] üzerinde
    def finalize(self): ...                # opsiyonel: kapanışta kaynak bırak
```

- Bir modül yalnız ihtiyaç duyduğu metodu uygular (hepsi opsiyonel; taban sınıf
  no-op sağlar).
- `setup()` worker kurulurken bir kez çağrılır (ağır kaynaklar burada).
- `finalize()` `CameraWorker.finalize()` içinde çağrılır (mediapipe/thread temizliği
  gibi; bu oturumda eklenen finalize disiplinine uyar).

## 4. Context Şeması (`ctx`)

Çekirdeğin doldurduğu alanlar (modüllerce **salt-okunur** kabul edilir):

| Anahtar | Tip | Açıklama |
|---------|-----|----------|
| `camera` | str | Kamera adı |
| `now` | float | `time.time()` (kare zaman damgası) |
| `detect_frame` | ndarray | Algılama uzayı karesi |
| `detect_dims` | (w, h) | Algılama uzayı boyutu (dw, dh) |
| `hires_frame` | ndarray | Yüksek çözünürlük kare |
| `hires_dims` | (w, h) | (hw, hh) |
| `scale` | (sx, sy) | detect → hires ölçek |
| `faces` | list[dict] | Bu karedeki yüzler (bbox detect uzayı, confidence, keypoints) |
| `tracks` | list[(tid, bbox)] | Aktif track'ler (detect uzayı) |
| `tid_face` | dict | `{track_id: face}` bu kare |
| `finished` | list[Track] | Bu karede biten görünümler (best-shot crop içerir) |
| `run_detect` | bool | Bu karede algılama çalıştı mı (modüller pahalı işi buna göre atlar) |

Çizim/çıktı:

| Anahtar | Tip | Açıklama |
|---------|-----|----------|
| `output` | ndarray | Display karesi. Başta `hires_frame`; `draw()` modülleri değiştirir. |

Modüller kendi çıktılarını `ctx`'e ekleyebilir (`embeddings`, `names`, `events`, ...).
Referanstaki `context` / `raw_frame` desenine karşılık gelir (`raw_frame` ≈ `hires_frame`).

## 5. Aşama-1 Modül Seti

Hepsi yeni `modules/` paketi altında; mevcut davranışı sarar (yeni özellik yok).

| Modül | Metod | Bugünkü karşılığı | Davranış |
|-------|-------|-------------------|----------|
| `RecognitionModule` | `process` | `_on_capture` (webui) | Web yakalama-sink'i: `ctx["finished"]` best-shot'larını RECENT'e yazar. `recognition_enabled` ise `RecognitionPipeline`'a submit eder (embedding + RECENT, async); değilse doğrudan `RECENT.add` (embedding'siz). Bugünkü `_on_capture` dallanmasının birebir karşılığı. |
| `CountingModule` | `process` | `worker._run_counting` | `ctx["tracks"]`'te çizgi-geçiş → `CountingStore` + async isim (`_NameResolver`) |
| `ZoomModule` | `draw` | `FrameTransformer.transform` | `ctx["output"]`'u en büyük yüze odaklı pan-zoom'lar |
| `OverlayModule` | `draw` | `_draw_overlay` | FPS + yüz durumu + kamera adı + ZOOM etiketi overlay'i |

Çekirdek, zincirden **sonra** `ctx["output"]`'u `output_size`'a resize eder
(bugünkü davranış). `RecognitionPipeline`, `CountingStore`, `LineCrossingCounter`,
`_NameResolver`, `FrameTransformer` **aynen** kullanılır; yalnız çağrı yerleri
modüllere taşınır.

### Sorumluluk taşımaları
- `CameraWorker.process()`'in `_run_counting` çağrısı → `CountingModule.process()`.
- Web yolunda `_on_capture`'ın RECENT'e yazan mantığı → `RecognitionModule.process()`
  (`ctx["finished"]` üzerinden). CLI yolunda (`main.py`) `_emit_capture`'ın
  `db.save_capture` dalı KORUNUR (o yolda modül zinciri kurulmayabilir / RECENT yok);
  yani core `collect_finished` + `_emit_capture(db)` primitifi durur, web'in RECENT
  sink'i modüle taşınır.
- Zoom bloğu (`transform`) → `ZoomModule.draw()`.
- `_draw_overlay` → `OverlayModule.draw()`.

## 6. Config Formatı ve Geriye-Uyum

```yaml
# Opsiyonel. YOKSA varsayılan zincir sentezlenir (bkz. aşağıda).
pipeline:
  - modules.recognition:RecognitionModule
  - modules.counting:CountingModule      # yalnız sayım kamerasında anlamlı
  - modules.zoom:ZoomModule
  - modules.overlay:OverlayModule
```

**Varsayılan zincir sentezi** (`pipeline:` yoksa) — bugünkü davranışı üretir:

1. `recognition_enabled` ise `RecognitionModule`.
2. Kamera sayım kamerası ise (`line_counter`/`counting_store` verilmiş) `CountingModule`.
3. `zoom_enabled` ise `ZoomModule`.
4. `debug_overlay` ise `OverlayModule`.

Böylece **mevcut `config.yaml`'lar değişmeden aynı çalışır.**

**Kamera-bazlı override:** kamera bloğunda `pipeline:` varsa o kamera için global
`pipeline:`/varsayılanın yerine geçer (kamera öncelikli). Bu, yalnız `pipeline`
alanına özgü açık bir kuraldır; diğer ayarların davranışını etkilemez.

## 7. Çizim Çakışması ve Sıra

- Yalnız `draw()` uygulayan modüller render eder.
- Çizim sırası = liste sırası (ör. `ZoomModule` önce, `OverlayModule` sonra).
- Referansın "tracker çizerse detector çizmez" kuralı artık **kompozisyonla**
  çözülür: çakışan iki çiziciyi aynı listeye koymazsın. Ekstra runtime kuralı yok.
- `ZoomModule` listede yoksa `ctx["output"]` ham `hires_frame` kalır (zoom kapalı
  davranışı).

## 8. Modül Yükleme ve Hata İzolasyonu

- **Yükleme:** `modules.recognition:RecognitionModule` biçiminde dotted-path;
  `importlib.import_module` + `getattr`. Yükleme sırasında birkaç constructor
  imzası denenir: `(config, camera)`, `(config)`, `()` (referanstaki desen).
- **Yükleme hatası:** modül atlanır + net WARNING log; zincir kalanla devam eder.
- **Çalışma hatası:** bir modülün `process()`/`draw()`'u exception atarsa
  **yakalanır, loglanır (throttled), o kare için o modül atlanır.** Zincir ve canlı
  akış çökmez. (Bugünkü `_PreviewWorker._loop` genel try/except'i modül düzeyine iner;
  böylece tek bir modül hatası tüm kamerayı düşürmez.)

## 9. Dosya/Modül Sınırları

- `pipeline.py` (yeni): `PipelineModule` taban sınıfı, `load_module(path, config, camera)`
  (importlib + imza denemeleri), `build_pipeline(config, camera, ...)` (liste veya
  varsayılan sentez), `Pipeline.run(ctx)` (iki geçiş + hata izolasyonu).
- `modules/` (yeni paket): `recognition.py`, `counting.py`, `zoom.py`, `overlay.py`.
- `worker.py`: `process()` iki faza ayrılır; çekirdek `ctx` doldurur, `self._pipeline.run(ctx)`
  çağırır; `__init__`'te `build_pipeline`; `finalize`'de modül `finalize()`'leri.
- Her dosya tek sorumluluk; büyüyen `worker.py` böylece küçülür.

## 10. Test Planı

- **Yükleyici:** geçerli dotted-path yükler; bozuk yol → atlanır + loglanır; imza
  denemeleri (3 varyant) doğru constructor'ı bulur.
- **Varsayılan zincir:** `pipeline:` yokken beklenen modül seti sentezlenir
  (recognition_enabled / sayım kamerası / zoom_enabled / debug_overlay kombinasyonları).
- **İzole modül:** her modülün `process()`/`draw()`'u fake `ctx` ile beklenen
  yan etkiyi yapar (RecognitionModule submit eder; CountingModule event üretir;
  ZoomModule `ctx["output"]`'u değiştirir; OverlayModule çizer).
- **Hata izolasyonu:** patlayan sahte modül zinciri kesmez; sonraki modül çalışır.
- **Entegrasyon:** varsayılan zincirle `CameraWorker.process()` eski davranışı verir
  (mevcut `test_worker_backend_selection` yeşil kalır; gerekirse genişletilir).

## 11. Riskler ve Azaltımlar

| Risk | Azaltım |
|------|---------|
| Refactor davranışı bozar | Davranış-koruma + varsayılan zincir; entegrasyon testi |
| Modül hatası kamerayı düşürür | Modül düzeyi try/except + throttled log |
| Çizim sırası/zoom regresyonu | Liste sırası açık; zoom yoksa ham frame; görsel doğrulama |
| Async tanıma/isim bozulur | Mevcut `RecognitionPipeline`/`_NameResolver` aynen kullanılır |
| `worker.finalize` modül temizliğini atlar | `finalize` modül `finalize()`'lerini çağırır (web yolu bu oturumda bağlandı) |

## 12. Kabul Kriterleri

1. `pipeline:` tanımsız mevcut `config.yaml` ile davranış birebir aynı (tanıma,
   sayım, zoom, overlay).
2. `pipeline:` listesiyle modül sırası/kümesi değiştirilebilir; kod değişmez.
3. Yeni bir modül = yeni dosya + config satırı (Aşama 2 alarm modülü bunu kanıtlar).
4. Bir modülün hatası kamerayı/zinciri düşürmez; loglanır.
5. Tüm mevcut testler yeşil + yeni modül/yükleyici testleri geçer.
