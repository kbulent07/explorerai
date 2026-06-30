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
