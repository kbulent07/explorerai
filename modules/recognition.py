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
        finished = ctx.get("finished") or []
        for tr in finished:
            # Rapor katmanina blackboard olayi (ReportingModule tuketir);
            # on_capture olmasa da uretilir (CLI'da da rapor calissin).
            ctx.setdefault("events", []).append({
                "type": "capture_finished", "camera": ctx.get("camera"),
                "ts": ctx.get("now"), "crop": tr.best_crop,
                "bbox": tr.best_bbox, "quality": tr.best_score,
            })
        if self.on_capture is None:
            return
        for tr in finished:
            self.on_capture(ctx.get("camera"), tr.best_crop, tr.best_bbox,
                            tr.best_score, tr.first_seen, tr.last_seen, tr.best_time)
