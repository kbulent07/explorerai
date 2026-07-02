# Faz A1 — ReportManager (REST Alarm/Rapor) Tasarımı

> **Tür:** Tasarım/spec. **Tarih:** 2026-07-02. **Faz:** A1 (bkz.
> `docs/superpowers/plans/2026-07-02-aieye-mimari-evrim-yol-haritasi.md`).
> **Ön koşul:** PR #1 (config-driven pipeline) merge edilmiş olmalı — `pipeline.py`,
> `modules/` paketi ve `ctx` sözlüğü bu tasarımın zeminidir. PR #2 (AiEye rename)
> merge'inden sonra uygulanır; bu belge yeni adlandırmayı (`aieye.*`, `AIEYE_*`) kullanır.

## 1. Amaç

AiEye olaylarını (giriş/çıkış sayımı, biten görünüm yakalamaları) uzak bir REST API'ye
güvenilir biçimde gönderen bir raporlama/alarm katmanı eklemek. YourEye referans
mimarisindeki `ReportManager` deseninin tek-node AiEye'a uyarlanması.

**Başarı ölçütü:** Ağ yokken sistem donmaz ve olay kaybolmaz (kuyruk + disk snapshot);
ağ gelince birikenler gönderilir; aynı olay cooldown içinde tekrarlanmaz; `reporting`
yapılandırılmamışsa sistem bugünkü gibi (hiçbir davranış değişikliği olmadan) çalışır.

## 2. Kapsam

**v1 kapsamda:**
- `reporting.py` — `ReportManager` (taşıma/güvenilirlik motoru, pipeline'dan habersiz).
- `modules/reporting.py` — `ReportingModule` (blackboard `ctx["events"]` tüketicisi).
- `CountingModule` ve `RecognitionModule`'e `ctx["events"]` **üretimi** (davranış-korumalı ek).
- Config: global `reporting:` bölümü + kamera-bazlı override (merge, kamera öncelikli).
- `api_key` mevcut `secrets_util.encrypt()/decrypt()` ile şifreli saklanır.
- HTTP: **stdlib `urllib.request`** (yeni bağımlılık yok).

**v1 kapsam DIŞI (YAGNI):**
- İsim çözümü sonrası ikinci "güncelleme" raporu (sayım event'i `name: null` gidebilir).
- MinIO/S3 görsel yükleme (Faz C5) — görsel base64 gömülü gönderilir.
- Alarm koşul motoru (hedef-isim listesi vb. FaceAlarm, Faz A2 sonrası).
- Web arayüzünde raporlama durumu ekranı.

## 3. Mimari — iki katman, blackboard akışı (Seçenek B)

```
CountingModule.process()      RecognitionModule.process()
  ctx["events"] += [...]        ctx["events"] += [...]
            \                        /
             ▼                      ▼
        ReportingModule.process(ctx)          (zincirde uretici modullerden SONRA)
          filtre: reporting.events
          report_manager.send(event)  ──►  ReportManager (daemon thread)
                                             rate-limit send() icinde uygulanir
                                             deque(500) ─► POST {gateway}/AiInput
                                             basarisiz ─► kuyrukta kalir + disk snapshot
```

- Modüller birbirini tanımaz; yalnız `ctx["events"]` anahtarı üzerinden konuşurlar
  (referans blackboard ilkesi). `ReportingModule` zincirde üreticilerden **sonra**
  yer almalıdır (process fazı liste sırasıyla çalışır — mevcut `Pipeline.run` garantisi).
- `ReportManager` pipeline'dan tamamen bağımsızdır; birim testte tek başına test edilir.

## 4. Blackboard sözleşmesi — `ctx["events"]`

Yeni context anahtarı. Üretici modül `ctx.setdefault("events", []).append(event)` yapar.
Event = düz dict; ortak alanlar + tipe özgü alanlar:

| Alan | Tip | Not |
|---|---|---|
| `type` | str | `"counting_crossing"` \| `"capture_finished"` |
| `camera` | str | kamera adı |
| `ts` | float | epoch saniye (`ctx["now"]`) |

**`counting_crossing`** (CountingModule): `direction` (`"in"`/`"out"`), `name`
(str \| None — asenkron çözüm öncesi çoğunlukla None), `jpeg` (bytes \| None — zaten
üretilen kırpıntı).

**`capture_finished`** (RecognitionModule): `crop` (ndarray BGR), `bbox`, `quality`.
(`ReportingModule` gönderim öncesi crop'u JPEG'e çevirir; ndarray REST'e gitmez.)

Üretici eklemeleri **davranış-korumalıdır**: mevcut `counting_store.record`/`on_capture`
çağrıları aynen kalır; event yazımı yalnız ektir. Mevcut testlerin assertion'ları değişmez.

## 5. `ReportManager` (reporting.py)

```python
class ReportManager:
    def __init__(self, config: dict): ...
    def send(self, event: dict) -> bool     # kuyruga al (rate-limit + boyut siniri); thread gonderir
    def can_send(self, key) -> bool         # durum DEGISTIRMEDEN cooldown kontrolu
    def stop(self): ...                     # kuyrugu diske yaz + thread'i kapat
def build_report_manager(config) -> ReportManager | None   # reporting.enabled degilse None
```

- **Kuyruk:** `collections.deque(maxlen=max_queue=500)` — dolunca en eski düşer (FIFO drop).
- **Gönderici:** daemon thread `_sender_loop`; kuyruk başındaki öğeyi POST eder; başarısızsa
  öğe **kuyrukta kalır**, `RETRY_DELAY=5s` bekleyip yeniden dener (sonsuz retry, referans deseni).
- **Rate-limit:** anahtar `(camera, type)`; `cooldown_seconds` (vars. 60) içinde ikinci olay
  **kuyruğa alınmaz** (sessizce düşer, debug loglanır); `once_per_day: true` ise gün başına 1.
  `can_send()` durum değiştirmeden kontrol; `send()` başarılı kuyruklamada state günceller.
- **Offline dayanıklılık:** POST hatasında kuyruk `queue_path` (vars. `report_queue.json`)
  dosyasına snapshot'lanır (görseller base64); başlangıçta dosya varsa yüklenip silinir.
  `stop()` da snapshot alır (kapanışta olay kaybolmaz).
- **HTTP:** `urllib.request.Request(gateway_base + "/AiInput", data=json, headers={"X-API-KEY": key,
  "Content-Type": "application/json"}, method="POST")`, `timeout=10`. 2xx = başarı.
- **Payload:** `{"camera", "branchId", "eventType", "triggeredAt" (ISO-8601), "direction"?,
  "name"?, "message", "image" (base64 JPEG | null)}`.
- **`api_key` çözümü:** `secrets_util.decrypt()` ile (`enc$...` destekli). Değer `enc$` ile
  başlıyor ama çözülemiyorsa (anahtar eksik/yanlış): **uyarı logla + reporting devre dışı**
  (yanlış anahtarla dış API'ye istek atılmaz). Web ayarlarında yazarken `secrets_util.encrypt()`.
- **Yapılandırılmamış durum:** `gateway_base` boş ya da `enabled: false` →
  `build_report_manager` None döner; hiçbir thread açılmaz.

**Kalite kuralı (referans §KALİTE):** HTTP asla üretici/pipeline thread'inde senkron çağrılmaz —
`send()` yalnız kuyruğa yazar, ağ işi daemon thread'te.

## 6. `ReportingModule` (modules/reporting.py)

- `setup(config, camera, services)`: `self.rm = services.get("report_manager")`;
  kamera-merge'li `reporting` config'inden `self.event_types` (izinli tipler).
- `process(ctx)`: `self.rm` None ise no-op. `ctx.get("events", [])` içinden tipi
  `event_types`'ta olanları `self.rm.send(event)`'e verir. `capture_finished` crop'unu
  JPEG'e encode eder (kalite 80). İşlenen event'ler **listeden çıkarılmaz** (başka tüketici
  modüller de okuyabilsin; liste her karede sıfırdan kurulur — ctx her `process()` çağrısında
  yeni oluşturulduğu için sızıntı yok).
- Modül hatası `Pipeline._run_one`'ın mevcut yakala/logla/atla mekanizmasıyla izole (değişiklik yok).

## 7. Config şeması

```yaml
# ---- raporlama / alarm (REST) ----
reporting:
  enabled: false
  gateway_base: ""            # or. "https://ornek.api.adresi" -> POST {base}/AiInput
  api_key: ""                 # enc$... (sifreli, secrets_util) veya duz metin
  branch_id: ""
  cooldown_seconds: 60        # (kamera, olay-tipi) basina en az bu araligla gonder
  once_per_day: false
  events: ["counting_crossing", "capture_finished"]
  queue_path: "report_queue.json"
  max_queue: 500
```

Kamera bloğunda opsiyonel `reporting:` override; sığ merge (kamera anahtarı globali ezer):
```yaml
cameras:
  - name: "Giris"
    detect_url: "..."
    reporting: { branch_id: "SUBE-2", events: ["counting_crossing"] }
```
Merge yardımcıı: `resolve_reporting(config, camera_cfg) -> dict` (`reporting.py` içinde).

## 8. Bağlama noktaları

- **Varsayılan zincir** (`pipeline._default_chain`): `reporting.enabled: true` ise zincirin
  SONUNA `"modules.reporting:ReportingModule"` eklenir (üreticilerden sonra olması garanti).
  Kullanıcı `pipeline:` listesi verirse sırayı kendi belirler (belgelenir).
- **`services` sözlüğü:** `worker.CameraWorker.__init__`'teki mevcut services'e
  `"report_manager"` eklenir. `ReportManager` **uygulama başına tek örnek**tir
  (kuyruk/rate-limit/dosya paylaşımı için) — kamera başına değil.
- **Kuruluş:** `build_report_manager(config)` hem `webui.py` (mevcut singleton kurulum
  bölgesi, `CountingStore` yanı) hem `main.py`'de çağrılır; `LiveManager`/`CameraWorker`'a
  parametreyle taşınır (mevcut `counting_store` taşınma deseniyle birebir aynı yol).
  Not: counting bugün yalnız webui yolunda kurulu; `ReportManager` bundan bağımsızdır —
  CLI'da da `capture_finished` raporları çalışır.
- **Kapanış:** `webui`/`main` kapanışında `report_manager.stop()` (snapshot + thread join).

## 9. Hata yönetimi özeti

| Durum | Davranış |
|---|---|
| `reporting` yok/kapalı | `build_report_manager` None; sıfır etki |
| Gateway erişilemez | Olay kuyrukta kalır; 5s retry; kuyruk diske snapshot |
| Kuyruk dolu (500) | En eski düşer (FIFO drop), throttled uyarı loglanır |
| `api_key` çözülemiyor (`enc$` + yanlış anahtar) | Uyarı + reporting devre dışı |
| Cooldown içinde tekrar olay | Kuyruğa alınmaz (debug log) |
| `ReportingModule.process` hatası | `Pipeline._run_one` yakalar; kare atlanır, akış sürer |
| Kapanış | `stop()`: kuyruk snapshot + thread join (timeout'lu) |

## 10. Test planı

**`tests/test_reporting.py`** (ReportManager, HTTP mock'lu — `urllib.request.urlopen`
monkeypatch):
1. `enabled: false` / `gateway_base` boş → `build_report_manager` None.
2. Cooldown: aynı `(camera, type)` ikinci olay kuyruğa alınmaz; farklı kamera alınır.
3. `once_per_day` davranışı.
4. FIFO drop: `max_queue` aşılınca en eski düşer.
5. Offline: POST hatası → `queue_path` dosyası oluşur; yeni instance dosyayı yükler ve siler.
6. Başarılı POST: doğru URL, `X-API-KEY` header, ISO-8601 `triggeredAt`, base64 görsel.
7. `api_key` `enc$` + anahtar yok → devre dışı (istek atılmaz).
8. `stop()`: snapshot yazılır, thread kapanır.

**`tests/test_module_reporting.py`** (ReportingModule, sahte ReportManager):
1. `events` filtreleme: yalnız izinli tipler `send`'e gider.
2. `report_manager` None → no-op (patlamaz).
3. `capture_finished` crop'u JPEG bytes'a çevrilir.

**Mevcut modül testleri** (`test_module_counting.py`, `test_module_recognition.py`):
`ctx["events"]` üretimi için ek assertion; mevcut assertion'lar aynen geçmeli
(davranış-koruma kanıtı). Tam suite yeşil kalmalı.

## 11. Kabul kriterleri

1. `reporting` tanımsız mevcut config ile tam suite geçer ve çalışma zamanı davranışı birebir aynıdır.
2. `enabled: true` + sahte gateway ile: sayım geçişi ve biten görünüm REST'e ulaşır (test 6).
3. Gateway kapalıyken sistem donmaz; olaylar `report_queue.json`'a düşer; yeniden başlatınca gönderilir.
4. `api_key` config'te `enc$...` olarak durur; log'larda/arayüzde düz görünmez.
