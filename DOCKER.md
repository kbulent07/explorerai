# FaceZoom — Docker ile Çalıştırma

Container, **headless** modda çalışır: yalnız `webui.py` (canlı izleme + bellek-içi
yakalama). Masaüstü pencere (`main.py`) container'da kullanılmaz.

## Gereksinimler
- Docker + Docker Compose
- RTSP kameralara ağ erişimi olan bir host

## Kurulum / Çalıştırma

```bash
# 1) İmajı derle (ilk seferde birkaç dakika: mediapipe/insightface derlenir)
docker compose build

# 2) Arka planda başlat
docker compose up -d

# 3) Logları izle (kamera bağlantıları, insightface yükleme)
docker compose logs -f
```

Arayüz:  **http://<host-ip>:5000**  (kullanıcı/parola `config.yaml > web`).

```bash
# Durdur
docker compose down

# Kod/ayar değişikliğinden sonra yeniden derleyip başlat
docker compose up -d --build
```

## Kalıcılık (volumes)
- `./config.yaml → /app/config.yaml` — ayarlar ve `/settings` ekranından
  eklenen/silinen kameralar kalıcı. (`config.yaml` yoksa önce `config.example.yaml`'ı
  kopyalayın; gerçek parolalar içerdiği için `.gitignore`'dadır.)
- `insightface-models → /home/appuser/.insightface` — ArcFace modeli (~300 MB)
  bir kez iner, sonraki açılışlarda yeniden inmez. (Container **non-root** uid 1000
  çalıştığı için model bu kullanıcının HOME'una iner; yol buna göre ayarlandı.)

## Güvenlik / Sağlık
- Container **non-root** (uid 1000) çalışır. Linux'ta `/settings`'in `config.yaml`'a
  yazabilmesi için host dosyası uid 1000'e yazılabilir olmalı (Docker Desktop
  Windows/Mac'te otomatik halledilir). Gerekirse `chmod 666 config.yaml`.
- **Web parolası:** `config.yaml > web > password` yerine `FACEZOOM_WEB_PASSWORD`
  ortam değişkeniyle verebilirsiniz (compose `environment:`). Parola boş **ve**
  `auth_enabled: true` ise tüm girişler reddedilir (kazara açık kalmasın diye).
- **Sağlık kontrolü:** İmajda `HEALTHCHECK` tanımlı; kimlik gerektirmeyen `/healthz`
  ucunu yoklar. `docker ps` STATUS sütununda `healthy`/`unhealthy` görünür.

## Ağ
- **Bridge (varsayılan):** Container LAN kameralarına **dışarı doğru** bağlanabilir;
  web arayüzü `-p 5000:5000` ile yayınlanır. Çoğu kurulum için yeterlidir.
- **Linux + host ağı (opsiyonel):** `docker-compose.yml` içindeki `network_mode: host`
  satırını açın — LAN kameralarına en doğrudan erişim. (Windows/Mac Docker
  Desktop'ta host ağı sınırlıdır, bridge kullanın.)

## Notlar
- `config.yaml > web > host: "0.0.0.0"` olmalı (container içinden dışarı yayın için).
- İmaj insightface derlemesi nedeniyle büyüktür; ilk build internet erişimi ister.
- CPU yükü yüksekse `config.yaml`'daki `preview_fps`, `detect_interval`,
  `detect_downscale`, `recognition_det_size` ayarlarını düşürün.

## GPU (NVIDIA/CUDA) ile çalıştırma

GPU'lu bir makinede kurulum CPU'dan **farklıdır**: farklı imaj (`Dockerfile.gpu`,
`onnxruntime-gpu` + CUDA tabanı) ve GPU'yu container'a açan bir compose override
kullanılır. YOLOX (kişi tespiti) ve insightface (yüz tanıma) CUDA'da koşar.

**Gereksinim (host):** NVIDIA GPU + güncel sürücü (`nvidia-smi` çalışmalı) +
Docker Desktop (WSL2 backend GPU'yu otomatik destekler). GPU'lu Linux host'ta
ayrıca **NVIDIA Container Toolkit** gerekir.

**Windows — tek tıkla (önerilen):** `setup_gpu_windows.bat`'a çift tıklayın.
Script GPU'yu doğrular, GPU imajını derler ve `config.yaml`'i **yüksek doğruluk**
için ayarlar:

- `compute_device: gpu` — CUDA sağlayıcısı (YOLOX + insightface)
- `cpu_profile: high` — algılama tam çözünürlükte + her karede (küçük/uzak yüzler)
- `recognition_det_size: 640` — yüz tanıma dedektörü en yüksek doğruluk

**Elle (herhangi bir platform):**

```bash
# config.yaml'da compute_device: gpu olduğundan emin olun, sonra:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# Durdur / loglar
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f
```

> **Doğruluk ayarı (GPU olmadan da):** `cpu_profile: high`, `recognition_det_size`
> ve **yüz tanıma modeli** (`recognition_model`) ayarlarını **Kamera Ayarları**
> ekranından da değiştirebilirsiniz. `high` profili CPU'da **ağırdır**; GPU'da
> önerilir. GPU yoksa GPU imajı gereksiz büyüktür, `setup.bat` (CPU) kullanın.
>
> **Yüz tanıma modeli:** `buffalo_l` (standart, ResNet50) varsayılandır.
> `antelopev2` (ResNet100, glint360k) daha **doğru** eşler; GPU'da önerilir, ilk
> seferde ~1 GB iner. Model değişince RAM'deki kimlikler sıfırlanır (eski/yeni
> embedding uzayları kıyaslanamaz). GPU kurulum scripti bunu sorup ayarlar.

## YOLOX modeli (yolox_person arka ucu için)

`detector_backend: yolox_person` kullanacaksanız `models/yolox_nano.onnx` dosyasını
imaja dahil edin veya bir volume ile bağlayın. `models/` klasörü `.dockerignore`'da
hariç tutulmuşsa, modeli runtime'da volume olarak mount edin:
`-v $(pwd)/models:/app/models`. CPU imajında `onnxruntime` (CPU) yeterlidir.
