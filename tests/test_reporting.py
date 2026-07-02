import base64
import json
import time
from datetime import datetime, timezone

import secrets_util
import reporting
from reporting import ReportManager, build_report_manager, resolve_reporting


def _rm(tmp_path, **over):
    cfg = {"gateway_base": "http://gw", "branch_id": "S1",
           "cooldown_seconds": 60, "once_per_day": False,
           "queue_path": str(tmp_path / "q.json"), "max_queue": 3}
    cfg.update(over)
    return ReportManager(cfg, api_key="KEY")


def test_build_none_when_disabled_or_empty(tmp_path):
    assert build_report_manager({}) is None
    assert build_report_manager({"reporting": {"enabled": False}}) is None
    assert build_report_manager(
        {"reporting": {"enabled": True, "gateway_base": ""}}) is None


def test_build_none_when_key_unresolved(tmp_path, monkeypatch):
    # sifreleme anahtari YOK -> enc$ cozulmez -> raporlama kapali
    monkeypatch.delenv("AIEYE_SECRET_KEY", raising=False)
    monkeypatch.delenv("AIEYE_KEY_FILE", raising=False)
    monkeypatch.setattr(secrets_util, "_cache", {"loaded": False, "fernet": None})
    cfg = {"reporting": {"enabled": True, "gateway_base": "http://gw",
                         "api_key": "enc$bozuktoken",
                         "queue_path": str(tmp_path / "q.json")}}
    assert build_report_manager(cfg) is None


def test_build_basarili_start_edilir(tmp_path):
    cfg = {"reporting": {"enabled": True, "gateway_base": "http://gw",
                         "api_key": "duzanahtar",
                         "queue_path": str(tmp_path / "q.json")}}
    rm = build_report_manager(cfg)
    assert rm is not None and rm._thread is not None
    rm.stop()


def test_gateway_bos_send_false(tmp_path):
    rm = _rm(tmp_path, gateway_base="")
    assert rm.send({"type": "t", "camera": "K", "ts": 1.0}) is False


def test_cooldown_ayni_anahtar_reddedilir(tmp_path):
    rm = _rm(tmp_path)
    e = {"type": "counting_crossing", "camera": "Kam", "ts": 1000.0}
    assert rm.send(e) is True
    assert rm.send({**e, "ts": 1030.0}) is False           # 60 sn dolmadi
    assert rm.send({**e, "ts": 1061.0}) is True            # doldu
    assert rm.send({**e, "camera": "Diger", "ts": 1030.0}) is True  # farkli anahtar


def test_once_per_day(tmp_path):
    rm = _rm(tmp_path, once_per_day=True, cooldown_seconds=0, max_queue=10)
    day1 = time.mktime((2026, 7, 2, 10, 0, 0, 0, 0, -1))
    assert rm.send({"type": "t", "camera": "K", "ts": day1}) is True
    assert rm.send({"type": "t", "camera": "K", "ts": day1 + 3600}) is False
    assert rm.send({"type": "t", "camera": "K", "ts": day1 + 86400}) is True


def test_fifo_drop_en_eski_duser(tmp_path):
    rm = _rm(tmp_path, cooldown_seconds=0, max_queue=3)
    for i in range(5):
        assert rm.send({"type": "t", "camera": f"K{i}", "ts": 1000.0 + i}) is True
    assert [p["camera"] for p in rm._queue] == ["K2", "K3", "K4"]


def test_resolve_reporting_kamera_oncelikli():
    cfg = {"reporting": {"branch_id": "G", "events": ["a"], "cooldown_seconds": 60}}
    cam = {"name": "K", "reporting": {"branch_id": "S2"}}
    m = resolve_reporting(cfg, cam)
    assert m["branch_id"] == "S2"
    assert m["events"] == ["a"]
    assert m["cooldown_seconds"] == 60
    assert resolve_reporting(cfg, None)["branch_id"] == "G"


class _Resp:
    status = 200
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_post_dogru_istek_kurar(tmp_path, monkeypatch):
    seen = {}
    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["key"] = req.get_header("X-api-key")
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["timeout"] = timeout
        return _Resp()
    monkeypatch.setattr(reporting.urllib.request, "urlopen", fake_urlopen)
    rm = _rm(tmp_path)
    rm.send({"type": "counting_crossing", "camera": "Kam", "ts": 1000.0,
             "direction": "in", "jpeg": b"JJ"})
    assert rm._post(rm._queue[0]) is True
    assert seen["url"] == "http://gw/AiInput"
    assert seen["key"] == "KEY"
    b = seen["body"]
    assert b["eventType"] == "counting_crossing"
    assert b["direction"] == "in"
    assert b["branchId"] == "S1"
    assert b["triggeredAt"] == datetime.fromtimestamp(
        1000.0, tz=timezone.utc).isoformat()
    assert base64.b64decode(b["image"]) == b"JJ"


def test_post_hata_false(tmp_path, monkeypatch):
    def fail(req, timeout=None):
        raise OSError("baglanti yok")
    monkeypatch.setattr(reporting.urllib.request, "urlopen", fail)
    rm = _rm(tmp_path)
    rm.send({"type": "t", "camera": "K", "ts": 1.0})
    assert rm._post(rm._queue[0]) is False


def test_offline_snapshot_ve_reload(tmp_path):
    rm = _rm(tmp_path, cooldown_seconds=0)
    rm.send({"type": "t", "camera": "K1", "ts": 1.0})
    rm.send({"type": "t", "camera": "K2", "ts": 2.0})
    rm._snapshot()
    qp = tmp_path / "q.json"
    assert qp.exists()
    rm2 = _rm(tmp_path)     # ayni queue_path -> yukler + dosyayi siler
    assert [p["camera"] for p in rm2._queue] == ["K1", "K2"]
    assert not qp.exists()


def test_sender_loop_gonderir_ve_bosaltir(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp())
    rm = _rm(tmp_path, cooldown_seconds=0)
    rm.send({"type": "t", "camera": "K", "ts": 1.0})
    rm.start()
    deadline = time.time() + 3.0
    while rm._queue and time.time() < deadline:
        time.sleep(0.05)
    rm.stop()
    assert not rm._queue


def test_stop_bekleyenleri_diske_yazar(tmp_path):
    rm = _rm(tmp_path, cooldown_seconds=0)
    rm.send({"type": "t", "camera": "K", "ts": 1.0})
    rm.stop()                 # thread baslamadi ama kuyruk dolu -> snapshot
    assert (tmp_path / "q.json").exists()


def test_basarili_drenaj_snapshot_dosyasini_siler(tmp_path, monkeypatch):
    # kesinti -> snapshot olusur; ag gelir, kuyruk bosalir -> dosya SILINMELI
    monkeypatch.setattr(reporting.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp())
    rm = _rm(tmp_path, cooldown_seconds=0)
    rm.send({"type": "t", "camera": "K", "ts": 1.0})
    rm._snapshot()                       # kesinti anini taklit et
    qp = tmp_path / "q.json"
    assert qp.exists()
    rm.start()
    deadline = time.time() + 3.0
    while rm._queue and time.time() < deadline:
        time.sleep(0.05)
    rm.stop()
    assert not rm._queue
    assert not qp.exists()               # bayat kopya kalmadi


def test_stop_bos_kuyrukta_bayat_dosyayi_siler(tmp_path):
    rm = _rm(tmp_path)
    (tmp_path / "q.json").write_text("[]", encoding="utf-8")
    rm.stop()
    assert not (tmp_path / "q.json").exists()


def test_bozuk_snapshot_kenara_alinir(tmp_path):
    (tmp_path / "q.json").write_text("BOZUK{{{", encoding="utf-8")
    rm = _rm(tmp_path)                   # acilista yukleme denenir
    assert not rm._queue                 # kuyruk bos kaldi, patlamadi
    assert not (tmp_path / "q.json").exists()
    assert (tmp_path / "q.json.bad").exists()
