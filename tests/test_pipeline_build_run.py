# tests/test_pipeline_build_run.py
import pytest

import pipeline


class _Rec(pipeline.PipelineModule):
    def __init__(self):
        self.calls = []

    def process(self, ctx):
        self.calls.append("p")
        ctx.setdefault("order", []).append("rec.process")

    def draw(self, ctx):
        ctx.setdefault("order", []).append("rec.draw")


class _Boom(pipeline.PipelineModule):
    def process(self, ctx):
        raise RuntimeError("patladim")

    def draw(self, ctx):
        raise RuntimeError("cizerken patladim")


def test_run_iki_gecis_sirasi():
    a, b = _Rec(), _Rec()
    p = pipeline.Pipeline([a, b])
    ctx = {}
    p.run(ctx)
    # once tum process (liste sirasi), sonra tum draw (liste sirasi)
    assert ctx["order"] == ["rec.process", "rec.process", "rec.draw", "rec.draw"]


def test_run_modul_hatasi_zinciri_kesmez():
    good = _Rec()
    p = pipeline.Pipeline([_Boom(), good])
    ctx = {}
    p.run(ctx)   # Boom patlar ama yakalanir
    assert "rec.process" in ctx["order"]   # good yine calisti


@pytest.mark.skip(reason="modules Task 3-6'da eklenecek; Task 6 sonrasi acilir")
def test_build_varsayilan_zincir_sentezi():
    cfg = {"zoom_enabled": True, "debug_overlay": False}
    p = pipeline.build_pipeline(cfg, "Kam", {}, is_counting=True)
    names = [type(m).__name__ for m in p.modules]
    assert names == ["RecognitionModule", "CountingModule", "ZoomModule"]


def test_build_acik_liste_kullanir():
    cfg = {"pipeline": ["tests.helpers_pipe:GoodModule"]}
    p = pipeline.build_pipeline(cfg, "Kam", {})
    assert [type(m).__name__ for m in p.modules] == ["GoodModule"]


def test_build_kamera_override_onceligi():
    cfg = {"pipeline": ["modules.overlay:OverlayModule"]}
    p = pipeline.build_pipeline(cfg, "Kam", {},
                                camera_pipeline=["tests.helpers_pipe:GoodModule"])
    assert [type(m).__name__ for m in p.modules] == ["GoodModule"]
