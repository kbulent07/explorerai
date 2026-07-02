# framing.py
# -----------------------------------------------------------------------------
# FaceTracker + FrameTransformer
#
# Bu modül, StageCam projesinin (Apple "Center Stage" benzeri yuze-duyarli
# kadraj takibi) cekirdek mantigi temel alinarak uyarlanmistir.
#
#   Orijinal proje : StageCam  -  https://github.com/K-Rutuparna1087/StageCam
#   Orijinal yazar : K Rutuparna
#   Lisans         : MIT
#
# MIT License (StageCam)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# -----------------------------------------------------------------------------
#
# Uyarlamalar (AiEye):
#   - FaceTracker artik bbox yaninda guven skoru ve MediaPipe landmark
#     (goz/burun/agiz) anahtar noktalarini da dondurur -> best-shot skorlamasi
#     ve frontallik olcumu icin gerekli.
#   - FrameTransformer; zoom_factor, smoothing ve "hold" (yuz kaybolunca geniss
#     kadraja yumuscak donus) gibi davranislarin config'ten ayarlanabilmesi icin
#     genisletildi. Acilis-bekleme (3 sn) kaldirildi; canli sistemde gereksiz.

import os
import time

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python import vision

# Bu mediapipe surumu yalnizca yeni "Tasks" API'sini sunar (eski mp.solutions
# yok). FaceDetector bir .tflite model dosyasi ister; varsayilan olarak proje
# icindeki models/ klasorunden okunur (kurulumda bir kez indirilir).
# full_range: uzak/kucuk ve egik (tepeden) yuzlerde short_range'den cok daha iyi.
# short_range yalniz ~2m mesafe + cepheden yuzler icindir.
_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "blaze_face_full_range.tflite",
)


class FaceTracker:
    """MediaPipe (Tasks API) tabanli yuz algilayici.

    detect() her yuz icin bir sozluk dondurur:
        {
            "bbox": (x, y, w, h),          # piksel cinsinden
            "confidence": float,           # algilama guveni 0..1
            "keypoints": {                 # piksel cinsinden anahtar noktalar
                "right_eye": (x, y),
                "left_eye": (x, y),
                "nose": (x, y),
                "mouth": (x, y),
                "right_ear": (x, y),
                "left_ear": (x, y),
            },
        }
    """

    # BlazeFace (short range) anahtar nokta sirasi
    _KP_NAMES = ["right_eye", "left_eye", "nose", "mouth", "right_ear", "left_ear"]

    def __init__(self, min_detection_confidence=0.6, model_path=None):
        model_path = model_path or _DEFAULT_MODEL
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Yuz algilama modeli bulunamadi: {model_path}\n"
                "Model dosyasini (varsayilan: models/blaze_face_full_range.tflite) "
                "indirin (bkz. README)."
            )
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame):
        """BGR kare al, yuz listesini dondur (yukaridaki sozluk formatinda)."""
        ih, iw = frame.shape[:2]
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        faces = []
        if not result.detections:
            return faces

        for det in result.detections:
            bb = det.bounding_box
            x, y, w, h = bb.origin_x, bb.origin_y, bb.width, bb.height

            # Kare sinirlarina kirp (negatif/tasar degerleri engelle)
            x = max(0, int(x))
            y = max(0, int(y))
            w = max(1, min(int(w), iw - x))
            h = max(1, min(int(h), ih - y))

            try:
                confidence = float(det.categories[0].score)
            except (IndexError, AttributeError, TypeError):
                confidence = 0.0

            keypoints = {}
            kps = det.keypoints or []
            for i, name in enumerate(self._KP_NAMES):
                if i < len(kps):
                    # Tasks API anahtar noktalari normalize (0..1) gelir
                    keypoints[name] = (int(kps[i].x * iw), int(kps[i].y * ih))

            faces.append(
                {
                    "bbox": (x, y, w, h),
                    "confidence": confidence,
                    "keypoints": keypoints,
                }
            )

        return faces

    def close(self):
        self.detector.close()


class FrameTransformer:
    """Yumuscak (titremesiz) dijital pan-zoom.

    StageCam'in EMA (ustel hareketli ortalama) yumuscatma fikri korunmustur:
    merkez ve zoom her karede hedefe dogru "smoothing" oraninda yaklasir, boylece
    kamera ani ziplamaz. AiEye'da:

      - Yuz VARKEN  : hedef merkez = secili yuz merkezi, hedef zoom = zoom_factor.
      - Yuz YOKKEN  : hold_seconds boyunca son konumu korur; sonra hedef zoom 1.0
                      (tam geniss kadraj) ve merkez kare ortasina yumuscakca doner.
    """

    def __init__(
        self,
        frame_width,
        frame_height,
        zoom_factor=2.5,
        smoothing=0.15,
        hold_seconds=1.5,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.zoom_factor = max(1.0, float(zoom_factor))
        self.smoothing = float(smoothing)
        self.hold_seconds = float(hold_seconds)

        # Anlik (yumuscatilmis) durum
        self.cur_cx = frame_width / 2.0
        self.cur_cy = frame_height / 2.0
        self.cur_zoom = 1.0

        # Yuzun en son goruldugu an
        self._last_face_time = None

    def _smooth(self, current, target):
        return current + (target - current) * self.smoothing

    def update_size(self, frame_width, frame_height):
        """Akis cozunurlugu degisirse merkez/sinirlar tutarli kalsin."""
        self.frame_width = frame_width
        self.frame_height = frame_height

    def transform(self, frame, target_bbox, now=None):
        """Kareyi hedef yuze gore yumuscakca zoomlayip dondurur.

        target_bbox: (x, y, w, h) veya None (yuz yok).
        Donus: (cikti_kare, zoomlu_mu)
        """
        now = time.time() if now is None else now
        fw, fh = self.frame_width, self.frame_height

        if target_bbox is not None:
            self._last_face_time = now
            x, y, w, h = target_bbox
            target_cx = x + w / 2.0
            target_cy = y + h / 2.0
            target_zoom = self.zoom_factor
        else:
            # Yuz yok: hold suresi dolana kadar mevcut kadraji koru
            held = (
                self._last_face_time is not None
                and (now - self._last_face_time) < self.hold_seconds
            )
            if held:
                target_cx, target_cy = self.cur_cx, self.cur_cy
                target_zoom = self.cur_zoom
            else:
                target_cx, target_cy = fw / 2.0, fh / 2.0
                target_zoom = 1.0

        # Yumuscak gecis
        self.cur_cx = self._smooth(self.cur_cx, target_cx)
        self.cur_cy = self._smooth(self.cur_cy, target_cy)
        self.cur_zoom = self._smooth(self.cur_zoom, target_zoom)

        zoomed = self.cur_zoom > 1.02
        if not zoomed:
            return frame, False

        zoom_w = int(fw / self.cur_zoom)
        zoom_h = int(fh / self.cur_zoom)

        x1 = int(self.cur_cx - zoom_w / 2)
        y1 = int(self.cur_cy - zoom_h / 2)
        # Pencereyi kare icinde tut
        x1 = max(0, min(x1, fw - zoom_w))
        y1 = max(0, min(y1, fh - zoom_h))
        x2 = x1 + zoom_w
        y2 = y1 + zoom_h

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return frame, False

        resized = cv.resize(cropped, (fw, fh), interpolation=cv.INTER_LINEAR)
        return resized, True
