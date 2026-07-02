# tests/test_pipeline_loader.py
import pipeline


class _Dummy(pipeline.PipelineModule):
    def __init__(self):
        self.setup_args = None

    def setup(self, config, camera, services):
        self.setup_args = (config, camera, services)


def test_base_metodlari_noop():
    m = pipeline.PipelineModule()
    # taban metodlar cagrilabilir ve hicbir sey dondurmez (no-op)
    assert m.process({}) is None
    assert m.draw({}) is None
    assert m.setup({}, "Kam", {}) is None
    assert m.finalize() is None


def test_load_module_yukler_ve_setup_cagirir():
    m = pipeline.load_module("tests.helpers_pipe:GoodModule",
                             {"k": 1}, "Kam", {"svc": 2})
    assert m is not None
    assert m.setup_args == ({"k": 1}, "Kam", {"svc": 2})


def test_load_module_bozuk_yol_none_doner():
    assert pipeline.load_module("yok.modul:Sinif", {}, "Kam", {}) is None
    assert pipeline.load_module("bozukbiciim", {}, "Kam", {}) is None
