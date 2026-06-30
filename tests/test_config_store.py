# tests/test_config_store.py
# -----------------------------------------------------------------------------
# config_store birim testleri: round-trip ekle/listele/sil, URL/host dogrulama
# (E3), ve mtime onbellek (C2). CONFIG_PATH gecici dosyaya yonlendirilir ki
# GERCEK config.yaml'a DOKUNULMASIN.
# -----------------------------------------------------------------------------

import os
import tempfile
import unittest

import config_store


class ConfigStoreTestBase(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("web:\n  port: 5000\ncameras: []\n")
        self._orig_path = config_store.CONFIG_PATH
        config_store.CONFIG_PATH = self.path
        config_store._cache = {"mtime": None, "data": None}  # onbellegi sifirla

    def tearDown(self):
        config_store.CONFIG_PATH = self._orig_path
        config_store._cache = {"mtime": None, "data": None}
        try:
            os.remove(self.path)
        except OSError:
            pass


class TestRoundTrip(ConfigStoreTestBase):
    def test_add_list_delete(self):
        config_store.add_camera("Giris", "rtsp://h/1", "rtsp://h/2")
        cams = config_store.list_cameras()
        self.assertEqual(len(cams), 1)
        self.assertEqual(cams[0][1]["name"], "Giris")
        self.assertEqual(cams[0][1]["hires_url"], "rtsp://h/2")

        removed = config_store.delete_camera(0)
        self.assertEqual(removed["name"], "Giris")
        self.assertEqual(len(config_store.list_cameras()), 0)

    def test_duplicate_name_rejected(self):
        config_store.add_camera("Cam", "rtsp://h/1")
        with self.assertRaises(ValueError):
            config_store.add_camera("Cam", "rtsp://h/9")

    def test_update_camera(self):
        config_store.add_camera("Eski", "rtsp://h/1")
        updated = config_store.update_camera(0, "Yeni", "rtsp://h/2")
        self.assertEqual(updated["name"], "Yeni")
        self.assertEqual(config_store.list_cameras()[0][1]["name"], "Yeni")

    def test_cache_reflects_writes(self):
        # Yazimdan sonra onbellek gecersiz olmali -> taze veri gorulmeli.
        self.assertEqual(len(config_store.list_cameras()), 0)
        config_store.add_camera("X", "rtsp://h/1")
        self.assertEqual(len(config_store.list_cameras()), 1)


class TestValidation(ConfigStoreTestBase):
    def test_build_hik_url_encodes_password(self):
        url = config_store.build_hik_url("10.0.0.5", "admin", "p@ss:1",
                                         channel=1, stream="main")
        self.assertIn("p%40ss%3A1", url)
        self.assertTrue(url.endswith("/Streaming/Channels/101"))

    def test_build_hik_url_rejects_bad_ip(self):
        for bad in ("10.0.0.5/x", "1.2.3.4 evil", "ho st", "a@b"):
            with self.assertRaises(ValueError):
                config_store.build_hik_url(bad, "u", "p", 1, "sub")

    def test_validate_url_requires_scheme(self):
        with self.assertRaises(ValueError):
            config_store.add_camera("C", "no-scheme-here")

    def test_validate_url_rejects_whitespace(self):
        with self.assertRaises(ValueError):
            config_store.add_camera("C", "rtsp://h/1 with space")


if __name__ == "__main__":
    unittest.main()
