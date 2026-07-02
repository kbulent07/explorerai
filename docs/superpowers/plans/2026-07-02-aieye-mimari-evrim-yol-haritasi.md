# AiEye — Mimari Evrim Yol Haritası (YourEye referansına göre)

> **Tür:** Planlama / yol haritası. **Tarih:** 2026-07-02.
> Girdi: YourEye CV sistemi mimari referansı (6 doküman, kaynak-doğrulanmış). Bu belge, mevcut
> AiEye (eski adı FaceZoom) sistemini bu referansa göre nasıl evrilteceğimizi fazlara böler.
> Önceki boşluk analizini (`docs/superpowers/specs/2026-07-02-referans-mimari-bosluk-analizi.md`)
> kaynak-doğrulanmış detayla günceller.

## Bağlam (neden)

Referans, `app.youreye.com.tr`'ye alarm gönderen, çok-işletmeli, self-healing bir production CV
gözetim sistemidir. Altı doküman okundu:
- `MIMARI_REFERANS.md` — kaynak-doğrulanmış tam mimari (3 katman, pool, RTSP, pipeline, raporlama, §10 zayıf noktalar).
- `DESEN_inference_pool.md` — merkezi GPU inference pool (ZMQ ROUTER/DEALER + shared memory).
- `DESEN_pipeline_moduler.md` — config-driven blackboard pipeline (`process(frame, context)`).
- `DESEN_rtsp_dayaniklilik.md` — self-healing RTSP (generation + feda edilebilir inner thread).
- `PROMPT_klonla.md` / `PROMPT_desenleri_uyarla.md` — yeniden kurma/uyarlama şablonları (yeni mimari bilgisi taşımaz).

**Proje bu turda AiEye olarak yeniden adlandırıldı** (rename dalı; ayrı PR).

## Referans mimarinin özü (4 çekirdek desen)

1. **Merkezi GPU inference pool:** Tek `model_pool` process; kameralar frame'i **shared memory**'ye
   yazar, ZMQ ile yalnız meta (`{model, frame_id, shm_name, shape, dtype}`) gönderir. Model başına
   kuyruk + batch worker + warm-up; identity başına max 1 istek; drop-tolerant (`SNDHWM/RCVHWM=5`,
   NOBLOCK). YOLO sabit shm (416²·3, batch), InsightFace ayrı socket + dinamik shm (tek tek).
2. **Self-healing 3 katman:** `pool_server` (bekçi, subprocess restart) → `run.py --business`
   (BusinessGroup) → kamera worker (1 process, 6 thread). Sinyal dosyaları (`cv_pool_ready`/`heartbeat`).
3. **RTSP dayanıklılığı:** `cap.read()` sonsuz bloklanmasına karşı **generation + feda edilebilir
   inner thread** (READ_TIMEOUT_S=5s, stale eleme, `cap.release()` yalnız sahip thread'de). Backoff
   `[5,10,30,90,300]`s, MAX_RETRY 3600s. Bounded queue (maxsize=2, en yeniyi tut). GStreamer düşük
   gecikme (`leaky=2 max-buffers=1 drop=true`), FFMPEG fallback.
4. **Config-driven blackboard pipeline:** `process(frame, context) -> context`; dotted-path
   importlib; `enabled`; çizim-sahipliği devri; `overlay()`; senkron/asenkron inference.

## Referansın "yapma" listesi (§10) → AiEye tasarım ilkeleri

Referans kendi zayıf noktalarını belgeliyor; bunları **kopyalamayacağız**, düzelteceğiz:

| Referans zayıf noktası (§10) | AiEye ilkesi |
|---|---|
| Düz-metin credential (config'te api_key/RTSP şifresi) | **Zaten çözülü:** `secrets_util` + `.aieye.key`/`AIEYE_SECRET_KEY`. Yeni REST api_key de buradan. |
| Hardcoded Linux yolu (cuDNN) | Yol/cihaz config + `platform_utils` soyutlaması. |
| Kapalı `batch_watchdog` (stall tespiti) | Heartbeat'i **gerçekten izleyen** watchdog aç (freeze-detect). |
| Kapalı `health._check_all()` | Kamera/process düzey sağlık ping'i açık gelsin. |
| PoolClient elle reconnect yok | Pool cevabı gelmezse `cv_pool_ready` mtime'a bakıp socket'i yenile. |
| Platform coupling (SIGTERM/shm/ipc Linux) | IPC→TCP düşüşü + tmp/shm/signal soyutlaması Windows'ta test edilsin. |
| Karışık `__pycache__` (py310+313) | Tek Python sürümü sabitle. |

## Yol haritası — fazlar

Her faz = ayrı `brainstorming → spec → plan → implement` döngüsü. Risk/değer sırasıyla.

### Faz 0 — Rebrand (TAMAMLANDI bu turda)
FaceZoom → AiEye (`refactor:` commit, ayrı dal + PR). Davranış-korumalı. **Straggler:**
`pipeline.py`, `modules/*` yalnız `feat/config-driven-pipeline`'da; PR #1 merge sonrası aynı üç
geçişli ikame ile süpürülecek.

### Faz A — Dış entegrasyon & kalıcılık (düşük risk, yüksek değer, tek-node korunur)
- **A1. ReportManager (REST alarm/rapor):** daemon `_sender_loop`, `deque(maxlen=500)` FIFO,
  `POST {gateway_base}/AiInput` header `X-API-KEY`, payload `{cameraId, moduleId, branchId,
  triggeredAt, mediaFolderPath, data, message}`. **Offline dayanıklılık:** `queue_log.json`
  snapshot+reload, `live_events.jsonl` (1MB trim). Rate-limit: `cooldown_seconds` + opsiyonel
  `once_per_day`, `can_send_alarm()`. Yeni `reporting.py` + pipeline `FaceAlarm` modülü.
  **KURAL (§KALİTE):** HTTP'yi sıcak yolda SENKRON çağırma → async kuyruk.
- **A2. Kalıcı kimlik DB + enrollment:** `enroll.py` (klasör=kişi → embedding ortalaması
  L2-normalize → `faces.pkl` = dict{isim→512-d}). Runtime: cosine + `recognition_similarity`
  (0.40) eşiği; altı "unknown". `recent.py` RAM store'un yanına kalıcı katman; motor (`buffalo_l`,
  `recognition.py`) hazır.
- **A3. FaceBlur (KVKK):** `faces` bbox anonimleştirme (gaussian/pixelate) draw modülü; `blur_enabled`.
- **A4. CwLogger/attendance:** JSONL attendance + `raw_frame`'den jpg yüz kırpma;
  cooldown/burst/pause spam koruması.

### Faz B — Algılama & sayım zenginleştirme (orta risk, izole, opt-in)
- **B1. ZoneCounter:** 4-nokta poligon + track **alt-orta** noktası; sıralı çift-çizgi (önce üst,
  sonra alt) = yön doğrulaması; sayım sonrası `face_wait_frames` içinde en yüksek skorlu yüz-isim
  eşleşmesi; `overlay()` ile statik UI. Mevcut `LineCrossingCounter` korunur.
- **B2. (Opsiyonel) YOLOv8 backend:** `detection_backend.py`'ye `yolov8_person` (+TensorRT `.engine`).
  YOLOX korunur; config seçer. Yalnız teknoloji yakınsaması isteniyorsa.
- **B3. Pipeline sözleşmesi zenginleştirme:** `overlay()`, async infer (`send`/`recv_noblock` +
  frame_id), çizim-sahipliği devrinin genelleştirilmesi. Mevcut `pipeline.py`'yi referans
  sözleşmeye yaklaştırır.

### Faz C — Dağıtık altyapı (yüksek risk, neredeyse yeniden-yazım; yalnız çok-kamera/çok-GPU ölçeğinde)
- **C1. GPU inference pool:** `model_pool` (ZMQ ROUTER, batch+warm-up, sinyal dosyaları, shm,
  drop-tolerant HWM, ipc→tcp fallback) + ince `PoolClient`.
- **C2. 3-katman self-healing:** `pool_server` bekçi + `BusinessGroup` supervisor + 6-thread worker.
- **C3. RTSP generation deseni** (mevcut `stimeout` çözümünün üstüne) + freeze-detect watchdog + GStreamer.
- **C4. Multi-tenant:** `businesses/<id>/config.yaml` + `config/system.yaml` + kamera-düzey merge; `run.py --business`.
- **C5. (Opsiyonel) MinIO/S3** (bağlantı yoksa no-op) + bağımsız shm `viewer.py` (MJPEG grid) + `hardware_monitor`.

## Risk / efor

| Faz | Efor | Risk | Not |
|-----|------|------|-----|
| 0 (rebrand) | Küçük | Düşük | Tamamlandı |
| A | Her biri ~1 spec→plan | Düşük | Davranış-korumalı, pipeline'a modül |
| B | Orta | Orta | İzole, opt-in |
| C | Büyük (C1+C2 çekirdek) | Yüksek | Mimari değişim; §10 zayıf noktalarını düzelterek kur |

## Öneri

Sıra: **Faz 0 (bitti) → A1 + A2 → A3/A4 → B → (gerekirse) C.** Faz A tek-node AiEye'ı referansın en
somut iş değerine (uzak alarm + kalıcı kimlik + KVKK bulanıklaştırma) taşır; mevcut config-driven
pipeline'a temiz modül olarak oturur. Faz C'yi ancak gerçek çok-node/çok-GPU ölçeği gerekince aç
(YAGNI) — ve girildiğinde §10 anti-desenlerini baştan düzelt.

## Sonraki adım

Rebrand PR'ı merge edildikten (ve PR #1 sonrası straggler süpürüldükten) sonra **A1 (ReportManager)**
için `brainstorming → spec → plan` döngüsünü başlat.
