# tests/helpers_pipe.py
# Pipeline yukleyici testleri icin yardimci modul sinifi.
import pipeline


class GoodModule(pipeline.PipelineModule):
    def __init__(self):
        self.setup_args = None

    def setup(self, config, camera, services):
        self.setup_args = (config, camera, services)
