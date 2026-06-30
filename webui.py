# webui.py
# -----------------------------------------------------------------------------
# Flask tabanli yerel galeri arayuzu.
#
# Yakalanan en-net yuz kareleri zaman damgasina gore (en yeni ustte) grid olarak
# listelenir. Tarih araligi ve kamera bazinda filtrelenebilir. Sadece YEREL AG
# icindir; basit HTTP Basic Auth (kullanici/parola) ile korunur.
#
# Calistirma:  python webui.py
# -----------------------------------------------------------------------------

import hmac
import logging
import os
import threading
import time
from functools import wraps

import cv2 as cv
import yaml
from flask import (
    Flask, Response, abort, jsonify, redirect, render_template_string, request,
    send_file, url_for,
)

import config_store
from live import LiveManager
from recent import RecentFaceStore
from recognition import FaceRecognizer, RecognitionPipeline

# webui.py bir GIRIS NOKTASIDIR; loglamayi kendisi kurar. (Onceden live->main
# zinciri uzerinden main.py'nin basicConfig'ini dolayli aliyordu; worker.py
# ayrimindan sonra o yan etki kalkti.)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# --- config (galeri/DB kaldirildi; her sey bellek-ici) ---
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

WEB = CONFIG.get("web", {})
log = logging.getLogger("facezoom.web")

# Web kimlik bilgileri: ORTAM DEGISKENI config'i ezer. Boylece parola hic
# dosyaya yazilmadan da verilebilir (config.yaml zaten .gitignore'da).
_WEB_USER = os.environ.get("FACEZOOM_WEB_USERNAME") or WEB.get("username", "admin")
_WEB_PASS = os.environ.get("FACEZOOM_WEB_PASSWORD") or WEB.get("password", "")
if WEB.get("auth_enabled", True) and not _WEB_PASS:
    log.warning("Web parolasi BOS! config.yaml web.password ayarlayin veya "
                "FACEZOOM_WEB_PASSWORD ortam degiskeni verin (auth_enabled=true).")
# Bellek butcesini RAM'e gore YAZILIM belirler: bostaki RAM'in bir orani,
# alt/ust sinirla kirpilir. Sabit kisi sayisi yerine bayt-butcesi: kucuk yuzler
# cok, buyuk yuzler az kayit tutar; RAM tassmaz. Dolunca en eski kayit dussar.
def _memory_budget():
    frac = float(CONFIG.get("recent_ram_fraction", 0.05))
    lo = int(CONFIG.get("recent_ram_min_mb", 64)) * 1024 * 1024
    hi = int(CONFIG.get("recent_ram_max_mb", 512)) * 1024 * 1024
    try:
        import psutil
        avail = psutil.virtual_memory().available
        return int(max(lo, min(avail * frac, hi)))
    except Exception:
        return lo

_BUDGET = _memory_budget()

# Bellek-ici son-yuzler deposu (DISKE YAZMAZ). Kisi kimligi yuz tanima
# (embedding kosinus) ile belirlenir; ayni kisi farkli konum/kamerada tek kayit.
RECENT = RecentFaceStore(
    best_window=CONFIG.get("recent_best_window_seconds", 120),
    max_bytes=_BUDGET,
    sim_threshold=CONFIG.get("recognition_similarity", 0.40),
)

# Yuz tanima pipeline'i (ayri thread + kuyruk). Diske kimlik yazmaz.
_RECOGNITION_ON = CONFIG.get("recognition_enabled", True)
RECOG_PIPE = None
if _RECOGNITION_ON:
    _recognizer = FaceRecognizer(
        model_name=CONFIG.get("recognition_model", "buffalo_l"),
        det_size=CONFIG.get("recognition_det_size", 320),
        min_det_score=CONFIG.get("recognition_min_det_score", 0.5),
    )
    # NOT: thread'i import aninda DEGIL, __main__'de start() ediyoruz (modulu
    # test/araclar icin import etmek agir thread baslatmasin). Yakalama yalniz
    # __main__ -> LIVE.start_all() sonrasi olur; o noktada pipeline calisiyor.
    RECOG_PIPE = RecognitionPipeline(
        _recognizer, RECENT,
        require_face=CONFIG.get("recognition_required", True),  # yuz yoksa saklama
    )


def _on_capture(camera_name, crop_bgr, bbox, quality, first_seen, last_seen, best_time):
    """CameraWorker bir gorunum bitirince cagrilir (canli thread'den).

    HIZLI olmali: tanima acikken kirpintiyi pipeline kuyruguna atar (agir is
    ayri thread'de). Tanima kapaliysa dogrudan (embedding'siz) belle.
    Hicbir durumda DISKE yazilmaz.
    """
    if RECOG_PIPE is not None:
        RECOG_PIPE.submit(camera_name, crop_bgr, bbox, quality, best_time)
    else:
        ok, buf = cv.imencode(".jpg", crop_bgr, [cv.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            RECENT.add(camera_name, bbox, buf.tobytes(), quality, ts=best_time)


# db=None -> DISKE/DB'ye yakalama YOK; en-net kareler yalniz bellekte (RECENT).
LIVE = LiveManager(CONFIG, db=None, on_capture=_on_capture)


app = Flask(__name__)


# --- basit kullanici/parola korumasi (HTTP Basic) ---
def check_auth(username, password):
    # SABIT-ZAMANLI karsilastirma (timing attack'a karsi). Bos parola = reddet.
    if not _WEB_PASS:
        return False
    user_ok = hmac.compare_digest(str(username or ""), str(_WEB_USER))
    pass_ok = hmac.compare_digest(str(password or ""), str(_WEB_PASS))
    return user_ok and pass_ok


def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not WEB.get("auth_enabled", True):   # yerel guvenli agda kapatilabilir
            return f(*args, **kwargs)
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            if auth:  # basarisiz giris denemesini logla (kaba denetim izi)
                log.warning("Basarisiz giris denemesi: kullanici=%r ip=%s",
                            auth.username, request.remote_addr)
            return Response(
                "Giriss gerekli", 401,
                {"WWW-Authenticate": 'Basic realm="FaceZoom"'},
            )
        return f(*args, **kwargs)
    return wrapper


@app.route("/")
@require_auth
def index():
    # Yuz galerisi kaldirildi (bellek-ici mod) -> dogrudan canli izlemeye yonlendir
    return redirect(url_for("watch"))


SETTINGS_PAGE = """
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FaceZoom &mdash; Kamera Ayarlari</title>
  <style>
    :root { color-scheme: dark; }
    body { font-family: system-ui, Arial, sans-serif; margin: 0;
           background: #14161a; color: #e8e8e8; }
    header { background: #1d2026; padding: 14px 20px; border-bottom: 1px solid #2a2e36; }
    h1 { margin: 0; font-size: 18px; }
    h2 { font-size: 15px; color: #cdd2da; margin: 22px 0 10px; }
    a { color: #6fa8ff; text-decoration: none; }
    .wrap { max-width: 820px; margin: 0 auto; padding: 18px; }
    .msg { padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; font-size: 14px; }
    .ok  { background: #163a26; border: 1px solid #2c6b46; color: #9be7bd; }
    .err { background: #3a1620; border: 1px solid #6b2c3a; color: #e79bb0; }
    table { width: 100%; border-collapse: collapse; background: #1d2026;
            border: 1px solid #2a2e36; border-radius: 8px; overflow: hidden; }
    th, td { text-align: left; padding: 9px 12px; font-size: 13px;
             border-bottom: 1px solid #2a2e36; word-break: break-all; }
    th { color: #9aa0aa; font-weight: 600; }
    .url { color: #9aa0aa; font-size: 12px; }
    form.add { background: #1d2026; border: 1px solid #2a2e36; border-radius: 10px;
               padding: 16px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 10px; }
    .field { display: flex; flex-direction: column; flex: 1; min-width: 140px; }
    label { font-size: 12px; color: #9aa0aa; margin-bottom: 4px; }
    input[type=text], input[type=password], input[type=number], select {
        background: #23262e; color: #e8e8e8; border: 1px solid #3a3f4a;
        border-radius: 6px; padding: 7px 9px; font-size: 14px; }
    .chk { display: flex; align-items: center; gap: 7px; font-size: 13px;
           color: #cdd2da; margin: 4px 0 12px; }
    button { background: #2f6df0; border: 1px solid #2f6df0; color: #fff;
             border-radius: 7px; padding: 8px 16px; font-size: 14px; cursor: pointer; }
    button.del { background: #b23a4a; border-color: #b23a4a; padding: 5px 11px; font-size: 12px; }
    button.edit { background: #2f8f5b; border-color: #2f8f5b; padding: 5px 11px; font-size: 12px; }
    button.cancel { background: #3a3f4a; border-color: #3a3f4a; }
    .editrow td { background: #191c22; }
    .editrow .row { margin-bottom: 8px; }
    details { margin: 6px 0 14px; }
    summary { cursor: pointer; color: #9aa0aa; font-size: 13px; }
    .hint { font-size: 12px; color: #7d838d; margin-top: 6px; }
    .note { font-size: 12px; color: #d8a657; margin-top: 14px; }
  </style>
</head>
<body>
  <header><h1>FaceZoom &mdash; Kamera Ayarlari
    <a href="{{ url_for('watch') }}" style="font-size:13px;font-weight:400;margin-left:12px;">&#128250; Canli Izleme</a></h1>
  </header>
  <div class="wrap">

    {% if message %}<div class="msg {{ 'ok' if ok else 'err' }}">{{ message }}</div>{% endif %}

    <h2>Tanimli Kameralar</h2>
    {% if cameras %}
    <table>
      <tr><th>#</th><th>Ad</th><th>Akislar</th><th></th></tr>
      {% for idx, cam in cameras %}
      <tr>
        <td>{{ idx + 1 }}</td>
        <td>{{ cam.get('name', '-') }}<br>
          <a href="{{ url_for('watch') }}" style="font-size:12px;">&#128250; izle</a></td>
        <td class="url">
          detect: {{ cam.get('detect_url', '-') }}<br>
          hires : {{ cam.get('hires_url', '(tek akis)') }}
        </td>
        <td style="white-space:nowrap;">
          <button class="edit" type="button" onclick="toggleEdit({{ idx }})">Duzenle</button>
          <form method="post" action="{{ url_for('settings_delete') }}" style="display:inline;"
                onsubmit="return confirm('Bu kamerayi sil?');">
            <input type="hidden" name="index" value="{{ idx }}">
            <button class="del" type="submit">Sil</button>
          </form>
        </td>
      </tr>
      <tr class="editrow" id="edit-{{ idx }}" style="display:none;">
        <td colspan="4">
          <form method="post" action="{{ url_for('settings_edit') }}">
            <input type="hidden" name="index" value="{{ idx }}">
            <div class="row">
              <div class="field"><label>Kamera adi</label>
                <input type="text" name="name" value="{{ cam.get('name','') }}" required></div>
            </div>
            <div class="row">
              <div class="field"><label>detect_url (sub)</label>
                <input type="text" name="detect_url" value="{{ cam.get('detect_url','') }}" required></div>
            </div>
            <div class="row">
              <div class="field"><label>hires_url (main &mdash; bos birakirsan tek akis)</label>
                <input type="text" name="hires_url" value="{{ cam.get('hires_url','') }}"></div>
            </div>
            <button type="submit">Kaydet</button>
            <button type="button" class="cancel" onclick="toggleEdit({{ idx }})">Vazgec</button>
            <div class="hint">RTSP URL'lerini dogrudan duzenle (paroladaki ozel karakter
              URL-encode olmali). Kaydedince bu kamera yeni ayarla yeniden baslar.</div>
          </form>
        </td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <p class="url">Henuz kamera tanimlanmamiss.</p>
    {% endif %}

    <h2>Yeni Hikvision Kamera Ekle</h2>
    <form class="add" method="post" action="{{ url_for('settings_add') }}">
      <div class="row">
        <div class="field"><label>Kamera adi</label>
          <input type="text" name="name" placeholder="Giris" required></div>
        <div class="field"><label>IP adresi (kamera veya NVR)</label>
          <input type="text" name="ip" placeholder="10.150.0.11" required></div>
        <div class="field" style="max-width:90px;"><label>Port</label>
          <input type="number" name="port" value="554"></div>
      </div>
      <div class="row">
        <div class="field"><label>Kullanici</label>
          <input type="text" name="username" value="admin"></div>
        <div class="field"><label>Parola</label>
          <input type="password" name="password" placeholder="kamera parolasi"></div>
        <div class="field" style="max-width:110px;"><label>Kanal</label>
          <input type="number" name="channel" value="1" min="1"></div>
      </div>
      <div class="hint">Dogrudan kameraya baglanirken kanal = 1 (101/102).
        NVR uzerinden ise kameranin kanal no.su (orn. 2 &rarr; 201/202).</div>
      <label class="chk"><input type="checkbox" name="single_stream"> Tek akis
        (yalniz sub-stream; ayri hires yok)</label>

      <details>
        <summary>Gelismiss: tam RTSP URL gir (Hikvision disi / ozel durum)</summary>
        <div class="row" style="margin-top:10px;">
          <div class="field"><label>detect_url (sub)</label>
            <input type="text" name="detect_url" placeholder="rtsp://..."></div>
        </div>
        <div class="row">
          <div class="field"><label>hires_url (main, opsiyonel)</label>
            <input type="text" name="hires_url" placeholder="rtsp://..."></div>
        </div>
        <div class="hint">Bu alanlar doldurulursa yukaridaki IP/kanal yerine
          dogrudan bu URL'ler kullanilir (parola zaten encode edilmiss olmali).</div>
      </details>

      <button type="submit">Kamera Ekle</button>
      <div class="note">&#9888; Degissiklikler config.yaml'a yazilir; calissan
        <code>main.py</code> bunlari yeniden baslatildiginda alir.</div>
    </form>
  </div>
  <script>
    function toggleEdit(i){
      var r = document.getElementById('edit-' + i);
      var open = (r.style.display !== 'none' && r.style.display !== '');
      r.style.display = open ? 'none' : 'table-row';
    }
  </script>
</body>
</html>
"""


@app.route("/settings")
@require_auth
def settings(message=None, ok=False):
    return render_template_string(
        SETTINGS_PAGE,
        cameras=config_store.list_cameras(),
        message=message, ok=ok,
    )


@app.route("/settings/add", methods=["POST"])
@require_auth
def settings_add():
    f = request.form
    try:
        manual_detect = (f.get("detect_url") or "").strip()
        if manual_detect:
            # Gelismiss mod: tam URL'ler
            config_store.add_camera(
                name=f.get("name"),
                detect_url=manual_detect,
                hires_url=f.get("hires_url"),
            )
        else:
            config_store.add_hik_camera(
                name=f.get("name"),
                ip=f.get("ip"),
                username=f.get("username") or "admin",
                password=f.get("password") or "",
                channel=int(f.get("channel") or 1),
                port=int(f.get("port") or 554),
                single_stream=bool(f.get("single_stream")),
            )
        return settings(message="Kamera eklendi.", ok=True)
    except (ValueError, TypeError) as e:
        return settings(message=f"Hata: {e}", ok=False)


@app.route("/settings/edit", methods=["POST"])
@require_auth
def settings_edit():
    f = request.form
    try:
        idx = int(f.get("index"))
        # Eski adi al -> worker'i durdurmak + eski kayitlari temizlemek icin.
        old_name = None
        for i, c in config_store.list_cameras():
            if i == idx:
                old_name = (c or {}).get("name")
                break
        updated = config_store.update_camera(
            idx,
            name=f.get("name"),
            detect_url=f.get("detect_url"),
            hires_url=f.get("hires_url"),
        )
        if not updated:
            return settings(message="Duzenlenecek kamera bulunamadi.", ok=False)
        # Ad VEYA URL degissmiss olabilir -> ilgili worker'lari durdur ki yeni
        # ayarla yeniden baslasinlar; eski/yeni ada ait bellek kayitlarini temizle.
        for nm in {old_name, updated.get("name")}:
            if nm:
                LIVE.stop_camera(nm)
                RECENT.drop_camera(nm)
        return settings(message=f"'{updated.get('name')}' guncellendi "
                                f"(kamera yeniden baslatildi).", ok=True)
    except (ValueError, TypeError) as e:
        return settings(message=f"Hata: {e}", ok=False)


@app.route("/settings/delete", methods=["POST"])
@require_auth
def settings_delete():
    try:
        idx = int(request.form.get("index"))
        removed = config_store.delete_camera(idx)
        if removed:
            # Config'ten cikarmak YETMEZ: calissan isleyici thread'ini durdur ve
            # bu kameranin bellekteki eski kayitlarini temizle; aksi halde kamera
            # yakalamaya devam eder ve resimler galeri/popup'ta gorunmeye devam eder.
            name = removed.get("name")
            if name:
                LIVE.stop_camera(name)
                dropped = RECENT.drop_camera(name)
                return settings(
                    message=f"'{name}' silindi (yakalama durduruldu, "
                            f"{dropped} kayit temizlendi).", ok=True)
            return settings(message="Kamera silindi.", ok=True)
        return settings(message="Silinecek kamera bulunamadi.", ok=False)
    except (ValueError, TypeError) as e:
        return settings(message=f"Hata: {e}", ok=False)


WATCH_PAGE = """
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FaceZoom &mdash; Canli Izleme</title>
  <style>
    :root { color-scheme: dark; }
    html, body { height: 100%; }
    body { font-family: system-ui, Arial, sans-serif; margin: 0; height: 100vh;
           display: flex; flex-direction: column; overflow: hidden;
           background: #14161a; color: #e8e8e8; }
    header { flex: 0 0 auto; background: #1d2026; padding: 12px 18px;
             border-bottom: 1px solid #2a2e36; }
    h1 { margin: 0; font-size: 17px; }
    a { color: #6fa8ff; text-decoration: none; }

    /* 3 sutun: %20 / %60 / %20 */
    .cols { flex: 1 1 auto; display: flex; min-height: 0; }
    .col { height: 100%; overflow-y: auto; box-sizing: border-box; }
    .col-cams { flex: 0 0 20%; max-width: 20%; border-right: 1px solid #2a2e36; padding: 10px; }
    .col-mid  { flex: 0 0 70%; max-width: 70%; padding: 12px;
                display: flex; flex-direction: row; gap: 12px; }
    .col-list { flex: 0 0 10%; max-width: 10%; border-left: 1px solid #2a2e36; padding: 8px; }
    .colhead { font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
               color: #7d838d; margin: 2px 4px 10px; position: sticky; top: 0;
               background: #14161a; padding: 4px 0; z-index: 2; }

    /* SOL: kameralar */
    .camitem { margin-bottom: 14px; }
    .camitem .t { font-size: 12px; font-weight: 600; margin-bottom: 5px;
                  display: flex; justify-content: space-between; }
    .camitem .t .cnt { color: #7d838d; font-weight: 400; }
    .livewrap { background: #0c0d10; border-radius: 8px; overflow: hidden; position: relative; }
    .livewrap img.live { width: 100%; display: block; background: #0c0d10;
                         object-fit: contain; cursor: zoom-in; }
    .livewrap:fullscreen { background: #000; display: flex; align-items: center; justify-content: center; }
    .livewrap:fullscreen img.live { max-height: 100vh; height: 100%; cursor: zoom-out; }
    .livewrap .pausedlabel { display: none; position: absolute; inset: 0;
        align-items: center; justify-content: center; color: #9aa0aa;
        font-size: 13px; background: #0c0d10; }
    .livewrap.paused .pausedlabel { display: flex; }
    .livewrap.paused img.live { visibility: hidden; }
    /* canli oynatma durdur/oynat dugmeleri */
    button.pp { background: #2a2e36; border: 1px solid #3a3f4a; color: #cdd2da;
                border-radius: 6px; padding: 2px 9px; font-size: 11px; cursor: pointer; }
    button.pp:hover { border-color: #3a86ff; }
    button.pp.allbtn { margin-left: 8px; }

    /* ORTA: DIKEY bolme -> SOL %70 buyuk resim, SAG %30 son 5 kisi (alt alta) */
    @keyframes popin { from { transform: scale(.92); opacity: 0; } to { transform: scale(1); opacity: 1; } }
    /* SOL %70: buyuk resim (spotlight) */
    #spot { flex: 50 1 0; min-width: 0; min-height: 0; display: flex; flex-direction: column;
            align-items: center; cursor: pointer; }
    #spot .colhead { align-self: stretch; }
    #spot.swap img { animation: popin .3s ease-out; }
    #spot img { flex: 1 1 auto; min-height: 0; width: 100%; height: 100%; object-fit: contain;
                border-radius: 10px; border: 1px solid #2f6df0; background: #0c0d10; }
    #spot .smeta { flex: 0 0 auto; margin-top: 10px; font-size: 15px; color: #cdd2da; text-align: center; }
    #spot .smeta .cam { color: #6fa8ff; font-weight: 600; }
    #spot .smeta .s { color: #6fcf97; }
    /* verilen isim (oturum-ici) */
    .nm { color: #ffd479; font-weight: 700; }
    #spot .shint { flex: 0 0 auto; margin-top: 4px; font-size: 12px; color: #7d838d; }

    /* SAG: son 10 kisi, 2 SUTUNLU grid (panel = sayfanin ~%20'si: 2x%10) */
    #last5wrap { flex: 20 1 0; min-width: 0; min-height: 0; display: flex; flex-direction: column;
                 border-left: 1px solid #2a2e36; padding-left: 12px; }
    .l5head { flex: 0 0 auto; font-size: 12px; text-transform: uppercase;
              letter-spacing: .04em; color: #7d838d; margin-bottom: 8px; }
    #last5 { flex: 1 1 auto; min-height: 0; display: grid; grid-template-columns: 1fr 1fr;
             gap: 8px; overflow-y: auto; align-content: start; }
    .l5 { min-height: 0; aspect-ratio: 3 / 4; display: flex; flex-direction: column; cursor: pointer;
          background: #23262e; border: 1px solid #2a2e36; border-radius: 8px; overflow: hidden; }
    .l5:hover { border-color: #3a86ff; }
    .l5.active { border-color: #2f6df0; box-shadow: 0 0 0 2px #2f6df0 inset; }
    .l5 img { flex: 1 1 auto; min-height: 0; width: 100%; object-fit: cover; background: #0c0d10; }
    .l5 .m { flex: 0 0 auto; font-size: 10px; padding: 3px 4px; color: #9aa0aa; text-align: center;
             overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }

    /* SAG: zaman sirali liste */
    .lrow { display: flex; gap: 8px; align-items: center; padding: 6px 4px;
            border-bottom: 1px solid #23262e; cursor: pointer; }
    .lrow:hover { background: #23262e; }
    .lrow.new { animation: fadein .35s ease-out; }
    @keyframes fadein { from { opacity: 0; } to { opacity: 1; } }
    .lrow img { width: 46px; height: 46px; object-fit: cover; border-radius: 6px;
                background: #0c0d10; flex: 0 0 auto; }
    .lrow .li { font-size: 11px; color: #cdd2da; min-width: 0; }
    .lrow .li .c { color: #6fa8ff; }
    .lrow .li .s { color: #6fcf97; }

    .placeholder { color: #7d838d; font-size: 12px; padding: 10px 4px; }
    .empty { padding: 60px; text-align: center; color: #9aa0aa; }
  </style>
</head>
<body>
  <header><h1>FaceZoom &mdash; Canli Izleme
    <a href="#" onclick="openPopup();return false;" style="font-size:13px;font-weight:400;margin-left:12px;">&#8599; Popup Pencere</a>
    <a href="{{ url_for('settings') }}" style="font-size:13px;font-weight:400;margin-left:12px;">Kamera Ayarlari</a>
  </h1></header>

  {% if names %}
  <div class="cols">
    <!-- SOL %20: kameralar -->
    <div class="col col-cams">
      <div class="colhead">Kameralar
        <button class="pp allbtn" type="button" id="allbtn" onclick="toggleAll()"
                title="Sadece gosterimi durdurur; yakalama arka planda devam eder">Tumunu Durdur</button>
        <button class="pp allbtn" type="button" id="allzoom" onclick="toggleZoomAll()"
                title="Tum kameralari saga-sola kaymadan tam kare goster">{{ 'Tumunu Normal' if zoom_default else 'Tumu Yuz Takibi' }}</button>
      </div>
      {% for n in names %}
      <div class="camitem">
        <div class="t"><span>{{ n }}</span>
          <span>
            <button class="pp" type="button" data-cam="{{ n }}" onclick="toggleCam(this)"
                    title="Sadece gosterimi durdurur; yakalama devam eder">Durdur</button>
            <button class="pp" type="button" data-zoom-cam="{{ n }}" onclick="toggleZoom(this)"
                    title="Saga-sola kaymadan tam kare goster">{{ 'Normal Goster' if zoom_default else 'Yuz Takibi' }}</button>
            <span class="cnt" data-cnt="{{ n }}">0 kisi</span>
          </span>
        </div>
        <div class="livewrap" ondblclick="toggleFs(this)" title="Cift tikla = tam ekran">
          <img class="live" data-name="{{ n }}" data-live="{{ url_for('live', name=n) }}"
               src="{{ url_for('live', name=n) }}" alt="{{ n }}">
          <div class="pausedlabel">&#9208; Duraklatildi</div>
        </div>
      </div>
      {% endfor %}
    </div>

    <!-- ORTA %60: DIKEY bolme -> solda buyuk resim (%70), sagda son 5 (%30) -->
    <div class="col col-mid">
      <div id="spot" onclick="resumeAuto()" title="Otomatik akisa donmek icin tikla">
        <div class="colhead" id="midhead">Son algilanan kisi</div>
        <img id="spotimg" alt="son algilanan kisi">
        <div class="smeta" id="spotmeta"></div>
        <div class="shint" id="shint"></div>
        <button class="pp" id="namebtn" type="button" style="margin-top:8px;"
                onclick="event.stopPropagation(); nameCurrent();">&#9998; Isim ver</button>
      </div>
      <div id="last5wrap">
        <div class="l5head">Son 8 kisi</div>
        <div id="last5"></div>
      </div>
    </div>

    <!-- SAG %20: zaman sirali liste -->
    <div class="col col-list">
      <div class="colhead">Zaman sirasi (en yeni ustte)</div>
      <div id="list"></div>
      <div class="placeholder" id="listph">Henuz kayit yok.</div>
    </div>
  </div>

  <script>
    function openPopup(){
      // 400x300 ayri pencere: solda en son kisi (300x300), sagda son 5 (100x300)
      window.open('{{ url_for("popup") }}', 'facezoom_popup',
        'width=400,height=300,menubar=no,toolbar=no,location=no,status=no,resizable=yes');
    }

    // --- canli oynatma durdur/oynat (istemci tarafi: MJPEG baglantisini kapat) ---
    const stoppedCams = new Set();
    function applyCam(img, play){
      const wrap = img.closest('.livewrap');
      if(play){
        // taze baglanti (cache-bust) -> donmus eski akisa takilma
        img.src = img.dataset.live + '?t=' + Date.now();
        if(wrap) wrap.classList.remove('paused');
      } else {
        img.removeAttribute('src');   // baglantiyi kapat -> sunucu thread'i serbest
        if(wrap) wrap.classList.add('paused');
      }
    }
    function setBtn(btn, stopped){ if(btn) btn.textContent = stopped ? 'Oynat' : 'Durdur'; }
    function syncAllBtn(){
      const imgs = document.querySelectorAll('img.live');
      const allStopped = imgs.length > 0 && [...imgs].every(im => stoppedCams.has(im.dataset.name));
      setBtnAll(allStopped);
    }
    function setBtnAll(allStopped){
      const b = document.getElementById('allbtn');
      if(b) b.textContent = allStopped ? 'Tumunu Oynat' : 'Tumunu Durdur';
    }
    function toggleCam(btn){
      const name = btn.dataset.cam;
      const img = document.querySelector('img.live[data-name="' + name + '"]');
      if(!img) return;
      const play = stoppedCams.has(name);     // su an duruyorsa -> oynat
      if(play) stoppedCams.delete(name); else stoppedCams.add(name);
      applyCam(img, play);
      setBtn(btn, !play);
      syncAllBtn();
    }
    function toggleAll(){
      const imgs = [...document.querySelectorAll('img.live')];
      // en az biri oynuyorsa hepsini durdur; hepsi duruyorsa hepsini oynat
      const anyPlaying = imgs.some(im => !stoppedCams.has(im.dataset.name));
      imgs.forEach(im => {
        const name = im.dataset.name;
        if(anyPlaying) stoppedCams.add(name); else stoppedCams.delete(name);
        applyCam(im, !anyPlaying);
      });
      document.querySelectorAll('button.pp[data-cam]').forEach(b =>
        setBtn(b, stoppedCams.has(b.dataset.cam)));
      setBtnAll(anyPlaying);
    }

    // --- canli zoom (dijital pan-zoom) ac/kapat: sunucu tarafinda worker'a uygula ---
    const zoomState = {};   // name -> zoom acik mi
    document.querySelectorAll('button.pp[data-zoom-cam]').forEach(b => {
      zoomState[b.dataset.zoomCam] = {{ 'true' if zoom_default else 'false' }};
    });
    function postZoom(name, enabled){
      const fd = new FormData();
      fd.append('name', name); fd.append('enabled', enabled ? '1' : '0');
      fetch('{{ url_for("zoom_toggle") }}', {method:'POST', body:fd});
    }
    // on = zoom (yuz takibi) acik mi -> dugme aksiyon etiketini gosterir
    function setZoomBtn(btn, on){ if(btn) btn.textContent = on ? 'Normal Goster' : 'Yuz Takibi'; }
    function toggleZoom(btn){
      const name = btn.dataset.zoomCam;
      const on = !zoomState[name];
      zoomState[name] = on;
      postZoom(name, on);
      setZoomBtn(btn, on);
      syncZoomAllBtn();
    }
    function toggleZoomAll(){
      const names = Object.keys(zoomState);
      const anyOn = names.some(n => zoomState[n]);  // biri aciksa hepsini kapat
      const target = !anyOn;
      names.forEach(n => zoomState[n] = target);
      postZoom('*', target);
      document.querySelectorAll('button.pp[data-zoom-cam]').forEach(b => setZoomBtn(b, target));
      syncZoomAllBtn();
    }
    function syncZoomAllBtn(){
      const names = Object.keys(zoomState);
      const allOff = names.length > 0 && names.every(n => !zoomState[n]);
      const b = document.getElementById('allzoom');
      if(b) b.textContent = allOff ? 'Tumu Yuz Takibi' : 'Tumunu Normal';
    }
    function toggleFs(el){
      if(document.fullscreenElement === el){ document.exitFullscreen(); }
      else if(el.requestFullscreen){ el.requestFullscreen(); }
      else if(el.webkitRequestFullscreen){ el.webkitRequestFullscreen(); }
    }
    function fmt(ts){ return new Date(ts*1000).toLocaleTimeString('tr-TR'); }
    function imgUrl(it){ return '/recent/'+it.id+'.jpg?t='+it.best_ts; }

    const listNodes = new Map();   // id -> sag liste satiri
    const l5Nodes = new Map();     // id -> orta-alt son 5 karti
    const itemsById = new Map();   // id -> son veri
    let firstPoll = true;

    // Spotlight (orta tek resim) durumu
    const MIN_SHOW_MS = 5000;      // bir resim en az bu kadar kalir
    let spotId = null;             // ortada gosterilen kisi
    let spotShownAt = 0;           // gosterildigi an (ms)
    let manualId = null;           // sagdan secilen (null = otomatik akis)
    let manualTimer = null;        // 20 sn tiklanmazsa otomatik akisa don
    const MANUAL_TIMEOUT_MS = 20000;

    // HTML kacisi: isim (/name ile KULLANICI girisi) ve kamera adi innerHTML'e
    // basildigi icin XSS'i onler. Sayisal alanlar (id/ts/quality) guvenli.
    function esc(s){ return String(s==null?'':s).replace(/[&<>"']/g, function(c){
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]; }); }
    function makeRow(it){
      const el = document.createElement('div');
      el.className = 'lrow' + (firstPoll ? '' : ' new');
      el.dataset.bestTs = it.best_ts;
      el.title = 'Ortada goster';
      el.onclick = () => manualShow(it.id);   // SAGDAN tikla -> ORTADA goster
      el.dataset.nm = it.name || '';
      el.innerHTML = '<img src="'+imgUrl(it)+'">'
        + '<div class="li">' + (it.name ? '<span class="nm">'+esc(it.name)+'</span><br>' : '')
        + '<span class="c">'+esc(it.camera)+'</span><br>'
        + fmt(it.best_ts)+'<br><span class="s">netlik '+it.quality+'</span></div>';
      if(!firstPoll) setTimeout(()=>el.classList.remove('new'), 3000);
      return el;
    }
    function refreshRow(el, it){
      // en-iyi kare (best_ts) VEYA isim degissince zaman metnini + resmi tazele
      if(el.dataset.bts == it.best_ts && el.dataset.nm == (it.name||'')) return;
      el.dataset.bts = it.best_ts; el.dataset.nm = it.name || '';
      el.querySelector('img').src = imgUrl(it);
      el.querySelector('.li').innerHTML = (it.name ? '<span class="nm">'+esc(it.name)+'</span><br>' : '')
        + '<span class="c">'+esc(it.camera)+'</span><br>'
        + fmt(it.best_ts)+'<br><span class="s">netlik '+it.quality+'</span>';
    }
    function makeL5(it){            // orta-alt: son N kisi karti
      const el = document.createElement('div');
      el.className = 'l5';
      el.dataset.bestTs = it.best_ts;
      el.dataset.nm = it.name || '';
      el.title = 'Ortada goster';
      el.onclick = () => manualShow(it.id);
      el.innerHTML = '<img src="'+imgUrl(it)+'"><div class="m">'+esc(it.name || it.camera)+'</div>';
      return el;
    }

    // --- ORTA tek resim (spotlight) ---
    function renderSpot(id, animate){
      // #spot HER ZAMAN gorunur (70/30 yapisi korunur); kisi yoksa placeholder.
      const it = itemsById.get(id);
      const spot = document.getElementById('spot');
      const img = document.getElementById('spotimg');
      const meta = document.getElementById('spotmeta');
      const hint = document.getElementById('shint');
      if(!it){
        img.style.visibility = 'hidden';
        meta.textContent = 'Henuz kisi algilanmadi';
        hint.textContent = '';
        return;
      }
      img.style.visibility = 'visible';
      const url = imgUrl(it);
      if(img.getAttribute('src') !== url) img.src = url;
      meta.innerHTML = (it.name ? '<span class="nm">'+esc(it.name)+'</span> &middot; ' : '')
        + '<span class="cam">'+esc(it.camera)+'</span> &middot; '
        + fmt(it.best_ts) + ' &middot; <span class="s">netlik '+it.quality+'</span>';
      hint.textContent = (manualId !== null)
        ? 'Secili kisi — resme tikla ya da 20 sn sonra otomatik akisa doner'
        : 'Otomatik: en son algilanan kisi (min 5 sn)';
      if(animate){ spot.classList.remove('swap'); void spot.offsetWidth; spot.classList.add('swap'); }
    }
    function setSpot(id){ spotId = id; spotShownAt = Date.now(); renderSpot(id, true); }

    // Ortada gosterilen kisiye isim ver (oturum-ici / RAM; embedding eslessince korunur)
    function nameCurrent(){
      const id = (manualId !== null) ? manualId : spotId;
      if(id === null || !itemsById.has(id)) return;
      const cur = itemsById.get(id);
      const name = prompt('Bu kisiye isim ver (bos = sil):', cur.name || '');
      if(name === null) return;                 // iptal
      const fd = new FormData(); fd.append('id', id); fd.append('name', name);
      fetch('/name', {method:'POST', body:fd}).then(() => {
        cur.name = name.trim() || null;          // anlik geri bildirim
        renderSpot(id, false);
      });
    }

    function manualShow(id){           // SAGDAN secim: o kisiyi sabitle
      manualId = id;
      document.getElementById('midhead').textContent = 'Secili kisi';
      renderSpot(id, true);
      // 20 sn tiklanmaz/secilmezse otomatik akisa don (her secimde sifirlanir)
      if(manualTimer) clearTimeout(manualTimer);
      manualTimer = setTimeout(resumeAuto, MANUAL_TIMEOUT_MS);
    }
    function resumeAuto(){             // ORTAYA tikla VEYA 20 sn sonra: akisa don
      if(manualTimer){ clearTimeout(manualTimer); manualTimer = null; }
      if(manualId === null) return;
      manualId = null;
      document.getElementById('midhead').textContent = 'Son algilanan kisi';
      spotShownAt = 0;                 // bir sonraki tick'te hemen en son algilanana gec
      tick();
    }

    // Spotlight zamanlayici: en son algilanani gosterir, min 5 sn'de bir gecer
    function tick(){
      if(manualId !== null){ renderSpot(manualId, false); return; }
      if(itemsById.size === 0){ renderSpot(null, false); return; }
      // en son algilanan = list_recent last_seen DESC -> ilk eleman
      const latest = currentItems[0];
      if(spotId === null || !itemsById.has(spotId)){ setSpot(latest.id); return; }
      if(latest.id !== spotId && (Date.now() - spotShownAt) >= MIN_SHOW_MS){
        setSpot(latest.id);
      } else {
        renderSpot(spotId, false);     // ayni kisi: en net kare guncellenmiss olabilir
      }
    }

    let currentItems = [];
    async function poll(){
      try {
        const r = await fetch('/recent.json', {cache:'no-store'});
        if(r.ok){
          const items = await r.json();           // last_seen DESC
          currentItems = items;
          const ids = new Set(items.map(i=>i.id));
          const list = document.getElementById('list');

          itemsById.clear();
          for(const it of items){ itemsById.set(it.id, it); }

          // kamera basi kisi sayaclari
          const counts = {};
          for(const it of items){ counts[it.camera] = (counts[it.camera]||0)+1; }
          document.querySelectorAll('[data-cnt]').forEach(c => {
            c.textContent = (counts[c.dataset.cnt]||0) + ' kisi';
          });

          // sag liste: en-iyi kare zamanina (best_ts) gore sirali, en yeni ustte
          const byBest = [...items].sort((a,b)=>b.best_ts-a.best_ts);
          for(const it of byBest){
            let lr = listNodes.get(it.id);
            if(!lr){ lr = makeRow(it); listNodes.set(it.id, lr); } else { refreshRow(lr, it); }
            list.appendChild(lr);
          }
          for(const [id, el] of [...listNodes]){ if(!ids.has(id)){ el.remove(); listNodes.delete(id); } }

          // ORTA-ALT: son 8 kisi (2 sutunlu grid)
          const last5 = items.slice(0, 8);
          const l5ids = new Set(last5.map(i=>i.id));
          const l5box = document.getElementById('last5');
          for(const it of last5){
            let el = l5Nodes.get(it.id);
            if(!el){ el = makeL5(it); l5Nodes.set(it.id, el); }
            else if(el.dataset.bestTs != it.best_ts || el.dataset.nm != (it.name||'')){
              el.dataset.bestTs = it.best_ts; el.dataset.nm = it.name || '';
              el.querySelector('img').src = imgUrl(it);
              el.querySelector('.m').textContent = it.name || it.camera;
            }
            l5box.appendChild(el);
          }
          for(const [id, el] of [...l5Nodes]){ if(!l5ids.has(id)){ el.remove(); l5Nodes.delete(id); } }
          for(const [id, el] of l5Nodes){ el.classList.toggle('active', id === spotId); }

          document.getElementById('listph').style.display = items.length ? 'none' : '';
          firstPoll = false;
        }
      } catch(e){ /* sessizce gec */ }
      setTimeout(poll, 1500);
    }
    poll();
    setInterval(tick, 500);   // orta spotlight zamanlayicisi (5 sn min gosterim)
  </script>
  {% else %}
  <div class="empty">Tanimli kamera yok. <a href="{{ url_for('settings') }}">Kamera ekleyin.</a></div>
  {% endif %}
</body>
</html>
"""


@app.route("/watch")
@require_auth
def watch():
    return render_template_string(
        WATCH_PAGE,
        names=LIVE.available_names(),
        zoom_default=bool(CONFIG.get("zoom_enabled", True)),
    )


POPUP_PAGE = """
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <title>FaceZoom &mdash; Popup</title>
  <style>
    :root { color-scheme: dark; }
    html, body { margin: 0; padding: 0; width: 400px; height: 300px; overflow: hidden;
                 background: #14161a; color: #e8e8e8;
                 font-family: system-ui, Arial, sans-serif; }
    #wrap { display: flex; width: 400px; height: 300px; }

    /* SOL 300x300: en son yakalanan kisi */
    #left { width: 300px; height: 300px; position: relative; background: #0c0d10;
            flex: 0 0 300px; }
    #leftimg { width: 300px; height: 300px; object-fit: contain; display: block; }
    #leftmeta { position: absolute; left: 0; right: 0; bottom: 0; padding: 3px 6px;
                font-size: 11px; background: rgba(0,0,0,.55); color: #cdd2da;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #leftmeta .cam { color: #6fa8ff; font-weight: 600; }
    #leftph { position: absolute; inset: 0; display: flex; align-items: center;
              justify-content: center; color: #7d838d; font-size: 12px; }

    /* SAG 100x300: son 5 kisi (alt alta, her biri 60px) */
    #right { width: 100px; height: 300px; flex: 0 0 100px; border-left: 1px solid #2a2e36;
             display: flex; flex-direction: column; }
    .r { flex: 1 1 0; min-height: 0; position: relative; border-bottom: 1px solid #23262e;
         background: #0c0d10; overflow: hidden; cursor: pointer; }
    .r:hover { outline: 2px solid #3a86ff; outline-offset: -2px; }
    .r img { width: 100%; height: 100%; object-fit: cover; display: block; }
    #leftmeta .pin { color: #6fcf97; }
    #leftmeta .nm { color: #ffd479; font-weight: 700; }
    .r .c { position: absolute; left: 0; bottom: 0; right: 0; font-size: 9px;
            padding: 1px 3px; background: rgba(0,0,0,.55); color: #6fa8ff;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .rph { flex: 1 1 0; display: flex; align-items: center; justify-content: center;
           color: #4a4f57; font-size: 11px; border-bottom: 1px solid #23262e; }
  </style>
</head>
<body>
  <div id="wrap">
    <div id="left">
      <img id="leftimg" alt="" style="visibility:hidden">
      <div id="leftph">Bekleniyor&hellip;</div>
      <div id="leftmeta"></div>
    </div>
    <div id="right"></div>
  </div>
  <script>
    var PIN_MS = 5000;            // sagdan tiklanan resim solda bu kadar sabit kalir
    var itemsById = {};
    var currentItems = [];
    var pinnedId = null;          // null = otomatik (en son kisi)
    var pinTimer = null;

    function imgUrl(it){ return '/recent/'+it.id+'.jpg?t='+it.best_ts; }
    function fmt(ts){ return new Date(ts*1000).toLocaleTimeString('tr-TR'); }

    function esc(s){ return String(s==null?'':s).replace(/[&<>"']/g, function(c){
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]; }); }
    function renderLeft(){
      var img = document.getElementById('leftimg');
      var ph  = document.getElementById('leftph');
      var meta= document.getElementById('leftmeta');
      // pinli kisi gecerliyse onu, degilse EN SON (currentItems[0]) goster
      var pinned = (pinnedId !== null && itemsById[pinnedId]);
      var it = pinned ? itemsById[pinnedId] : currentItems[0];
      if(!it){ img.style.visibility='hidden'; ph.style.display='flex'; meta.textContent=''; return; }
      var u = imgUrl(it);
      if(img.getAttribute('src') !== u) img.src = u;
      img.style.visibility = 'visible';
      ph.style.display = 'none';
      meta.innerHTML = (it.name ? '<span class="nm">'+esc(it.name)+'</span> &middot; ' : '')
        + '<span class="cam">'+esc(it.camera)+'</span> &middot; '+fmt(it.best_ts)
        + (pinned ? ' &middot; <span class="pin">secili</span>' : '');
    }

    function pin(id){              // sagdan tikla -> solda buyut, 5 sn sonra en sona don
      pinnedId = id;
      renderLeft();
      if(pinTimer) clearTimeout(pinTimer);
      pinTimer = setTimeout(function(){ pinnedId = null; renderLeft(); }, PIN_MS);
    }

    async function poll(){
      try {
        const r = await fetch('/recent.json', {cache:'no-store'});
        if(r.ok){
          const items = await r.json();   // last_seen DESC -> [0] en son
          currentItems = items;
          itemsById = {};
          for(const it of items){ itemsById[it.id] = it; }
          renderLeft();
          const last5 = items.slice(0, 5);
          const box = document.getElementById('right');
          if(last5.length){
            box.innerHTML = last5.map(function(it){
              return '<div class="r" onclick="pin('+it.id+')" title="Buyut (5 sn)">'
                   + '<img src="'+imgUrl(it)+'"><div class="c">'+esc(it.name || it.camera)+'</div></div>';
            }).join('');
          } else {
            box.innerHTML = '<div class="rph">son 5</div>';
          }
        }
      } catch(e){ /* sessizce gec */ }
      setTimeout(poll, 1500);
    }
    poll();
  </script>
</body>
</html>
"""


@app.route("/popup")
@require_auth
def popup():
    return render_template_string(POPUP_PAGE)


# Kamera basina eszamanli MJPEG izleyici sayaci. Her akis bir waitress thread'i
# tutar; sinirsiz izleyici (orn. cok sekme) thread havuzunu (web.threads) tuketip
# poll/json/diger kameralari ac birakabilir. Bu yuzden kamera basina sinir koyulur.
_stream_counts = {}
_stream_lock = threading.Lock()
_MAX_STREAMS = int(WEB.get("max_streams_per_camera", 8))


def _mjpeg_stream(name):
    """Kamera icin MJPEG (multipart/x-mixed-replace) akisi ureten generator.

    Tarayici uyumlulugu icin her parcada Content-Length basligi gonderilir ve
    standart sinir (boundary) cerceveleme kullanilir. Kare henuz hazir degilse
    SONSUZA KADAR bekleriz (akisi kapatmayiz) -> <img> bozuk gorunmez, kamera
    baglanir baglanmaz goruntu akmaya baslar.
    """
    # Worker'i bir kez al; sonra dogrudan onun JPEG'ini oku. (Onceden her karede
    # LIVE.get_jpeg -> ensure -> manager kilidi cagriliyordu; her istemci x kamera
    # her karede tek global kilidi cekistirip poll/json isteklerini geciktiriyordu.)
    pw = LIVE.ensure(name)
    if pw is None:
        return
    # Eszamanli izleyici sinirini uygula (thread tukenmesini onle).
    with _stream_lock:
        if _stream_counts.get(name, 0) >= _MAX_STREAMS:
            log.warning("Kamera '%s' icin eszamanli izleyici siniri (%d) doldu; "
                        "yeni akis reddedildi.", name, _MAX_STREAMS)
            return
        _stream_counts[name] = _stream_counts.get(name, 0) + 1
    try:
        last_sent = None
        while True:
            jpeg = pw.get_jpeg()
            if jpeg is None:
                time.sleep(0.1)
                continue
            # Ayni kareyi tekrar gondermeyelim (bant genisligi); yine de canli tut
            if jpeg is last_sent:
                time.sleep(0.03)
                continue
            last_sent = jpeg
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                b"\r\n" + jpeg + b"\r\n"
            )
            time.sleep(0.04)  # ~25 fps ust siniri
    finally:
        # Istemci kopunca (GeneratorExit) veya akis bitince sayaci dussur.
        with _stream_lock:
            _stream_counts[name] = max(0, _stream_counts.get(name, 1) - 1)


@app.route("/live/<name>")
@require_auth
def live(name):
    if name not in LIVE.available_names():
        abort(404)
    # Kamerayi hemen baslat (ilk kare ~1-2 sn icinde gelir)
    LIVE.ensure(name)
    resp = Response(
        _mjpeg_stream(name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    # NOT: 'Connection' hop-by-hop basligi WSGI'de (waitress) yasaktir -> koyma.
    resp.headers["X-Accel-Buffering"] = "no"  # ara proxy tamponlamasini kapat
    return resp


@app.route("/zoom", methods=["POST"])
@require_auth
def zoom_toggle():
    """Canli dijital pan-zoom'u ac/kapat. name='*' -> tum kameralar."""
    name = request.form.get("name") or "*"
    enabled = request.form.get("enabled") == "1"
    if name == "*":
        LIVE.set_zoom_all(enabled)
    else:
        LIVE.set_zoom(name, enabled)
    return ("", 204)


@app.route("/name", methods=["POST"])
@require_auth
def set_name():
    """Bir kimlige (RAM'de) isim ver. Kisi tekrar gelince embedding ile eslessip
    ismi korunur. Diske YAZILMAZ."""
    try:
        eid = int(request.form.get("id"))
    except (TypeError, ValueError):
        return ("gecersiz id", 400)
    ok = RECENT.set_name(eid, request.form.get("name", ""))
    return ("", 204) if ok else ("bulunamadi", 404)


@app.route("/recent.json")
@require_auth
def recent_json():
    """Son penceredeki (varsayilan 2 dk) kisiler; en yeni ustte. Bellekten."""
    return jsonify(RECENT.list_recent())


@app.route("/recent/<int:eid>.jpg")
@require_auth
def recent_image(eid):
    """Bellekteki en-net yuz JPEG'i (diskte degil)."""
    jpeg = RECENT.get_jpeg(eid)
    if jpeg is None:
        abort(404)
    resp = Response(jpeg, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/healthz")
def healthz():
    """Saglik kontrolu (Docker healthcheck). KIMLIK GEREKTIRMEZ; hassas veri
    dondurmez (yalniz sayisal durum)."""
    try:
        st = RECENT.stats()
        with _stream_lock:
            active_streams = sum(_stream_counts.values())
        return jsonify({
            "status": "ok",
            "cameras": len(LIVE.available_names()),
            "active_streams": active_streams,
            "recent_count": st.get("count", 0),
            "recent_bytes": st.get("bytes", 0),
        })
    except Exception:
        return jsonify({"status": "error"}), 500


if __name__ == "__main__":
    host = WEB.get("host", "0.0.0.0")
    port = int(WEB.get("port", 5000))
    # MJPEG akislari uzun omurludur ve thread tutar; bol thread ayir ki canli
    # yayinlar galeri/poll isteklerini bloke etmesin.
    threads = int(WEB.get("threads", 64))
    print(f"FaceZoom galeri:  http://{host}:{port}/  (kullanici: {_WEB_USER})")
    print(f"Bellek butcesi (RAM'e gore): {_BUDGET // (1024*1024)} MB  "
          f"(~{_BUDGET // 30000} kisi tahmini)")
    # Tek-ornek nobeti: main.py de yakalama yapar; ayni anda ikisi calisirsa
    # kameralar cift acilir. Bloke etmez, yalniz uyarir.
    import caplock
    caplock.acquire()
    # Tanima pipeline thread'ini KAMERALARDAN ONCE baslat (import aninda degil;
    # boylece modul testte import edilince agir thread baslamaz).
    if RECOG_PIPE is not None:
        RECOG_PIPE.start()
    # Tum kameralari ac: surekli yakalama (best-shot -> galeri) + canli yayin.
    LIVE.start_all()
    try:
        # Production WSGI server (dayanikli; werkzeug dev server degil)
        from waitress import serve
        print(f"waitress ile sunuluyor (threads={threads})")
        serve(app, host=host, port=port, threads=threads,
              channel_timeout=300, ident="FaceZoom")
    except ImportError:
        print("waitress yok; werkzeug dev server'a dusuluyor")
        app.run(host=host, port=port, debug=False, threaded=True)
