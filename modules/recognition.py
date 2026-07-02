# modules/recognition.py
# Web yakalama-sink modulu: biten gorunumlerin (ctx['finished']) best-shot'larini
# uygulama on_capture geri-cagrimina iletir. on_capture (webui) tanima acikken
# RecognitionPipeline'a submit eder, degilse dogrudan RECENT'e yazar -> mevcut
# _on_capture dallanmasi webui'de aynen korunur. CLI'da on_capture None -> no-op.

from pipeline import PipelineModule


class RecognitionModule(PipelineModule):
    def setup(self, config, camera, services):
        self.on_capture = services.get("on_capture")

    def process(self, ctx):
        if self.on_capture is None:
            return
        for tr in ctx.get("finished") or []:
            self.on_capture(ctx.get("camera"), tr.best_crop, tr.best_bbox,
                            tr.best_score, tr.first_seen, tr.last_seen, tr.best_time)
