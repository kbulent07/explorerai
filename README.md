# FaceZoom

Sabit (PTZ olmayan) Hikvision RTSP kameralarda yüz algılayıp canlı **dijital
zoom** (kırp + büyüt, Apple "Center Stage" benzeri yumuşak/titremesiz) yapan;
her "görünüm" (bir kişinin kameraya girip çıkışı) için **en net yüz karesini**
zaman damgasıyla **SQLite** veritabanına kaydeden ve basit bir **web galeride**
listeleyen uygulama.

- Yüz **tanıma yoktur**. Her görünüm ayrı kayıttır.
- Tüm bağımlılıklar açık kaynak, **bulut yok**, veritabanı sunucusuz (SQLite).
- Kod RTSP URL'lerini config'den olduğu gibi okur; IP'nin **kamera mı NVR mı**
  olduğunu bilmek zorunda değildir.

## Çekirdek: StageCam

Yüz takibi ve yumuşak pan-zoom motoru
[StageCam](https://github.com/K-Rutuparna1087/StageCam) (MIT, K Rutuparna)
projesinden uyarlanmıştır. İlgili `FaceTracker` + `FrameTransformer` mantığı
[`framing.py`](framing.py) içine **kopyalanıp** (MIT lisans başlığı korunarak)
ihtiyaçlarımıza göre genişletilmiştir (landmark/güven çıktısı, yapılandırılabilir
zoom/yumuşatma/hold). StageCam bir bağımlılık olarak **kurulmaz**.

## Mimari

| Dosya | Görev |
|-------|-------|
| [`framing.py`](framing.py) | StageCam'den uyarlanan `FaceTracker` (MediaPipe) + `FrameTransformer` (yumuşak dijital zoom) |
| [`camera.py`](camera.py) | Thread'li RTSP okuyucu + otomatik yeniden bağlanma; detect (sub) + hires (main) ayrı akış |
| [`tracking.py`](tracking.py) | Kareler arası `track_id` ataması (IoU/merkez) + best-shot (en net kare) skorlaması |
| [`db.py`](db.py) | SQLite şema, yazma, retention (KVKK) temizliği |
| [`main.py`](main.py) | Çoklu kamera canlı zoom döngüsü, overlay, kısayollar |
| [`webui.py`](webui.py) | Flask galeri arayüzü |
| [`config.yaml`](config.yaml) | Tüm ayarlar (kamera URL'leri koda gömülmez) |

### Çift akış (dual-stream) mantığı
- Yüz **algılama** düşük çözünürlüklü **sub-stream** (`detect_url`) üzerinde
  yapılır → düşük CPU.
- Canlı zoom görüntüsü ve kaydedilecek en net yüz kırpıntısı yüksek çözünürlüklü
  **main-stream** (`hires_url`) üzerinden alınır → yüksek çözünürlüklü foto.
- detect bbox'u sub çözünürlükten hires çözünürlüğüne oranlanır.
- `hires_url` verilmezse `detect_url` her ikisi için kullanılır (tek akış modu).

## Kurulum

`mediapipe` Python **3.10–3.12** ister (3.13/3.14 desteklenmez). Bu projede
**Python 3.12** ile kurulum doğrulanmıştır.

```powershell
# Windows (PowerShell) — Python 3.12 ile sanal ortam:
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux/macOS:
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Python 3.12 kurulu değilse: Windows'ta `py install 3.12`, ya da
> [python.org](https://www.python.org/downloads/) üzerinden indirin.

> `requirements.txt` sürümleri **sabitlenmiştir** (doğrulanmış çalışan ortam).
> mediapipe/numpy/insightface zinciri sürüme duyarlı olduğundan güncellemeden
> önce izole bir ortamda deneyin.

### Yapılandırma dosyası (zorunlu)

`config.yaml` gerçek kamera parolaları + web parolası içerdiğinden sürüm kontrolü
**dışındadır** (`.gitignore`). İlk kurulumda şablondan oluşturun:

```bash
cp config.example.yaml config.yaml      # Windows: copy config.example.yaml config.yaml
```

Sonra `config.yaml`'da `web > password` değerini **güçlü bir parola** yapın (veya
`FACEZOOM_WEB_PASSWORD` ortam değişkeniyle verin) ve kameralarınızı girin.

### Yüz algılama modeli (zorunlu)

Bu `mediapipe` sürümü yeni **Tasks API**'sini kullanır ve bir model dosyası
ister. İki seçenek var (`config.yaml` → `face_model` ile seçilir):

- **`blaze_face_full_range.tflite`** (~1 MB) — **varsayılan**. Uzak, küçük ve
  eğik/tepeden görünen yüzlerde belirgin şekilde daha iyi (güvenlik kamerası,
  NVR, geniş açı için önerilir).
- **`blaze_face_short_range.tflite`** (~230 KB) — yalnız ~2 m mesafe + cepheden
  yüzler; daha hızlı, yakın çekim/masaüstü için.

```powershell
# Windows (PowerShell) — her ikisini de indir
mkdir models -Force
Invoke-WebRequest -UseBasicParsing -OutFile "models\blaze_face_full_range.tflite" `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
Invoke-WebRequest -UseBasicParsing -OutFile "models\blaze_face_short_range.tflite" `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
```

```bash
# Linux/macOS
mkdir -p models
curl -L -o models/blaze_face_full_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite
curl -L -o models/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
```

> Modeller yalnızca kurulumda bir kez indirilir; çalışma anında internet/bulut
> kullanılmaz.

#### Tepeden / uzak kameralar için ipuçları

Yüzler sub-stream'de (640×360) çok küçük kaldığında algılama zayıflar. `config.yaml`:

- `detect_on_hires: true` — algılamayı yüksek çözünürlüklü (hires) akışta yapar
  (biraz daha CPU; tepeden geniş açı kameralarda şart).
- `detection_confidence: 0.4` ve `min_face_size: 20` — daha çok (uzak) yüz yakalar.
- En iyi yüz fotoğrafları **göz hizası** kameralardan (örn. giriş kapısı) gelir;
  tavandan tepeden bakan kameralar izleme için iyidir ama yüz açısı düşüktür.

## Hikvision RTSP URL formatı

```
rtsp://KULLANICI:PAROLA@IP:554/Streaming/Channels/{kanal}{akış}
```

- kanal N **main** stream → `{N}01` (101, 201, 301 …) — yüksek çözünürlük (`hires_url`)
- kanal N **sub**  stream → `{N}02` (102, 202, 302 …) — düşük çözünürlük (`detect_url`)

**Doğrudan kameraya** bağlanırken kanal hep 1 → `101` / `102`:
```yaml
- name: "Giris"
  detect_url: "rtsp://admin:Parola%40123@10.150.0.11:554/Streaming/Channels/102"
  hires_url:  "rtsp://admin:Parola%40123@10.150.0.11:554/Streaming/Channels/101"
```

**NVR üzerinden** bağlanırken kanal numarası kameraya göre değişir → `2xx`:
```yaml
- name: "Uretim"
  detect_url: "rtsp://admin:Parola%40123@10.150.0.50:554/Streaming/Channels/202"
  hires_url:  "rtsp://admin:Parola%40123@10.150.0.50:554/Streaming/Channels/201"
```

### Parola URL-encode
Parolada özel karakter varsa URL-encode edin: `@` → `%40`, `:` → `%3A`,
`/` → `%2F`, `#` → `%23` …

### TCP transport (önemli)
Hikvision UDP'de bozuk kare / paket kaybı yapabilir. Kod, RTSP transport'unu
**TCP**'ye zorlar — `cv2.VideoCapture` açılmadan önce ortam değişkeni ayarlanır:
```python
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
```
Bu, [`camera.py`](camera.py) içinde otomatik yapılır.

## Yapılandırma

Tüm ayarlar [`config.yaml`](config.yaml) içindedir. Öne çıkanlar:

| Anahtar | Açıklama |
|---------|----------|
| `detection_confidence` | MediaPipe min algılama güveni |
| `min_face_size` | Bundan küçük yüzleri ele (yanlış pozitif) |
| `detect_interval` | Kaç karede bir algılama (performans) |
| `zoom_factor` | Dijital zoom oranı |
| `smoothing` | Yumuşatma (küçük = daha yumuşak/yavaş) |
| `hold_seconds` | Yüz kaybolunca geniş kadraja dönmeden bekleme |
| `track_timeout` | Bu süre yüz görülmezse görünüm biter → kayıt |
| `track_center_dist_factor` | Kareler arası merkez-mesafe eşleşme eşiği (algılama genişliğinin oranı; vars. 0.2) |
| `quality_weights` | Best-shot skor ağırlıkları (netlik/boyut/frontallik/pozlama/güven) |
| `retention_days` | KVKK: bu süreden eski kayıt + dosya silinir |
| `webcam_test_index` | `0` verilirse RTSP yerine webcam (test) |
| `show_windows` | Canlı zoom pencereleri |
| `debug_overlay` | FPS + yüz durumu + kamera adı overlay |
| `web.username` / `web.password` | Web erişim bilgileri (env `FACEZOOM_WEB_USERNAME`/`FACEZOOM_WEB_PASSWORD` **ezer**) |
| `web.auth_enabled` | `false` → parola kapalı (yalnız güvenli yerel ağ) |
| `web.max_streams_per_camera` | Kamera başına eşzamanlı canlı izleyici sınırı (vars. 8; thread koruması) |
| `web.threads` | waitress worker thread sayısı (vars. 64) |
| kamera `role` | `giris` / `cikis` → `/sayim` giriş-çıkış sayımı (kimlik bazlı) |

## Çalıştırma

### Canlı uygulama (zoom + kayıt)
```bash
python main.py
```
- config'deki her kamera için ayrı pencere açılır.
- Yüz yokken tam geniş kare; yüz varken o yüze yumuşak dijital zoom.
- Bir kişi kameradan çıkınca (`track_timeout`) o görünümün **tek ve en net**
  karesi diske + DB'ye yazılır.

**Kısayollar:** `q` çıkış · `s` anlık snapshot · `f` tam ekran aç/kapat

### Test (kamerasız, webcam ile)
`config.yaml` içinde:
```yaml
webcam_test_index: 0
```
ayarlayıp `python main.py` çalıştırın — RTSP olmadan yerel webcam'le çalışır.

### Web galeri
```bash
python webui.py
```
Tarayıcıdan `http://<makine-ip>:5000/` adresini açın. En net yüz küçük resimleri
zaman damgasına göre (en yeni üstte) grid olarak listelenir; **tarih aralığı** ve
**kamera** bazında filtrelenebilir. Basit kullanıcı adı/parola (`config.yaml` →
`web:`) ile korunur.

> Galeri yalnızca **yerel ağ** içindir; internete açmayın. Varsayılan parolayı
> mutlaka değiştirin (`config.yaml > web > password` veya `FACEZOOM_WEB_PASSWORD`).
> HTTP Basic **düz metindir**; güvensiz ağda TLS için ters-proxy (nginx/Caddy)
> arkasına koyun. Parola **boş** ve `auth_enabled: true` ise tüm girişler reddedilir.
> Sağlık kontrolü: kimliksiz `GET /healthz` (Docker healthcheck kullanır).

### Kamera ayarları ekranı (web)

Galeri başlığındaki **⚙ Kamera Ayarları** bağlantısı (`/settings`) ile kamera
ekleme/silme arayüzü açılır. Aynı kullanıcı adı/parola (`config.yaml` → `web:`,
varsayılan `admin` / `123456`) ile korunur.

- **Hikvision modu:** kamera adı, IP, kullanıcı, parola ve kanal girilir; RTSP
  URL'leri otomatik kurulur (kanal N → main `N01` / sub `N02`), paroladaki özel
  karakterler URL-encode edilir. "Tek akış" seçeneği yalnız sub-stream kullanır.
  - Doğrudan kameraya bağlanırken **kanal = 1**.
  - NVR üzerinden ise kameranın kanal numarası (örn. **2** → 201/202).
- **Gelişmiş mod:** Hikvision dışı / özel durumlar için tam `detect_url` /
  `hires_url` doğrudan girilebilir.

Değişiklikler [`config.yaml`](config.yaml)'a yorumları **bozmadan** yazılır
(ruamel.yaml). Çalışan `python main.py`, yeni kameraları **yeniden başlatınca**
alır.

## Canlı izleme + yüz tanıma (web, bellek-içi)

`webui.py` çalışırken `/watch` sayfası 3 sütunludur: **sol %20** canlı kameralar,
**orta %60** kişi yüzleri (grid), **sağ %20** zaman sıralı liste. Sağdaki bir yüze
tıklayınca ortada büyük gösterilir; ortadaki resme tıklayınca akan listeye dönülür.

- **Yüz tanıma** (`insightface` / ArcFace embedding) ile aynı kişi, konum ve kamera
  bağımsız tek kayda toplanır (`recognition_*` ayarları). Tanıma ağır olduğu için
  yalnız "en-net kare" kırpıntısında, **ayrı bir thread/kuyrukta** çalışır; canlı
  zoom akışını yavaşlatmaz.
- Her kişinin gösterilen karesi, **son `recent_best_window_seconds`** (2 dk) içindeki
  **en net** karedir. Kamera başına `recent_per_camera_limit` (50) kişi tutulur.
- **Bu modda hiçbir şey diske yazılmaz** — yüz görüntüleri ve kimlik (embedding)
  yalnızca **bellekte** tutulur (uygulama kapanınca silinir). insightface modeli
  (`buffalo_l`) ilk kullanımda `~/.insightface/models` altına bir kez iner; bu
  model dosyasıdır, kişi/kimlik verisi değildir.

### Giriş/çıkış sayımı (`/sayim`)

Kameralara **rol** atanırsa (Kamera Ayarları → `giris` / `cikis`) `/sayim` sayfası
içeri giren / dışarı çıkan **kişi** sayısını ve isim bazlı listeleri gösterir.
Sayım **kimlik bazlıdır** (yüz tanıma): aynı kişi tekrar görülse de bir kez sayılır.
Bir kişiye `/watch` üzerinden RAM'de isim verilebilir (diske yazılmaz).

> Sayım verisi **bellektedir**: uygulama yeniden başlayınca sıfırlanır; RAM bütçesi
> dolunca en eski kimlikler düşeceğinden sayılar zamanla azalabilir — kalıcı/kümülatif
> bir geçiş sayacı **değildir**. Doğruluğu, aynı kişinin giriş ve çıkış kameralarında
> yüz tanımayla eşleşmesine bağlıdır.

## Dayanıklılık & performans

- **Production WSGI (waitress):** `python webui.py` artık dev server yerine
  **waitress** ile sunar (uzun süreli/çok izleyicili kullanımda dayanıklı). Eşzamanlı
  MJPEG akışları için `web.threads` (varsayılan 64) thread ayrılır.
- **RTSP donma koruması:** TCP transport'a `stimeout;5000000` (5 sn) eklendi —
  akış donarsa `read()` sonsuz bloke olmaz, otomatik yeniden bağlanma tetiklenir.
- **CPU ayarları** (çoklu kamera için): `preview_fps` (web önizleme/işleme hızı,
  varsayılan 12) ve `detect_downscale` (algılamayı bu oranda küçültülmüş karede yap,
  varsayılan 0.5). Bu ikisi kamera başına CPU'yu kabaca yarıya indirir.
- **Erişim:** `web.auth_enabled: false` ile (yalnız güvenli yerel ağda) parola
  tamamen kapatılabilir; aksi halde HTTP Basic auth (düz metin — internete açmayın).

## Veri ve gizlilik (KVKK)

> Aşağıdaki disk kaydı, **yalnız `main.py`** (masaüstü mod) için geçerlidir.
> Web (`webui.py`) modu varsayılan olarak diske yazmaz (yukarıya bakın).

- `main.py`: yüz görüntüleri `images_dir` (varsayılan `captures/`) altında tarih
  klasörlerine `.jpg` olarak yazılır; veritabanında yalnızca **dosya yolu** tutulur.
- `retention_days`'ten eski kayıtlar ve dosyaları periyodik olarak (her
  `cleanup_interval_hours` saatte bir) otomatik silinir.
- Bu sistem biyometrik nitelikte görüntü işler; kurulum ve saklama sürelerini
  yürürlükteki KVKK yükümlülüklerinize göre yapılandırın.

## Geliştirme / testler

Birim testleri ağır bağımlılık (mediapipe/insightface) gerektirmez, hızlı çalışır:

```powershell
venv\Scripts\python -m unittest discover -s tests      # Windows
```
```bash
python -m unittest discover -s tests                   # Linux/macOS
```

Kapsam: takip eşleştirmesi (IoU + belirleyici eşleşme), best-shot kalite skoru,
`config_store` round-trip + URL/host doğrulama, `RecentFaceStore` (embedding
eşleşme + bayt-bütçesi tahliyesi).

## Lisans / atıf

Yüz takibi ve pan-zoom çekirdeği StageCam (MIT, K Rutuparna) projesinden
uyarlanmıştır; MIT lisans başlığı [`framing.py`](framing.py) içinde korunmuştur.
