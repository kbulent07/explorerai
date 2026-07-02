# AiEye - Docker imaji
# -----------------------------------------------------------------------------
# Container YALNIZ webui.py'yi calistirir: canli izleme + bellek-ici yakalama
# (DISKE/DB'ye yazmaz). main.py cv2 ile masaustu pencere actigi icin headless
# container'a UYGUN DEGILDIR; bu yuzden container'da kullanilmaz.
#
# Build : docker compose build
# Calistir: docker compose up -d
# -----------------------------------------------------------------------------

FROM python:3.12-slim

# --- sistem bagimliliklari ---
#  ffmpeg            : OpenCV'nin RTSP (H.264) akislari cozmesi icin
#  libgl1, libglib2  : opencv-python yuklenirken bu .so'lar gerekir (GUI olmasa da)
#  libgomp1          : onnxruntime (OpenMP runtime)
#  build-essential,
#  cmake             : insightface'in C++/Cython eklentilerini derlemek icin
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Once derleme yardimcilari: insightface sdist'i build sirasinda numpy/Cython ister.
RUN pip install --no-cache-dir numpy Cython

# Bagimliliklar ayri katmanda (kod degisince pip yeniden calismasin -> hizli build)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# mediapipe Tasks API'nin C bindings'i calisma aninda libEGL.so.1 / libGLESv2.so.2
# ister. Bunlari pip KATMANINDAN SONRA ayri (hizli) bir katmanda kuruyoruz ki
# pahali pip onbellegi gecersiz olmasin.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 \
        libgles2 \
    && rm -rf /var/lib/apt/lists/*

# Uygulama kodu + yuz algilama modelleri (models/*.tflite)
COPY . .

# --- non-root kullanici (defense-in-depth: container root olarak calismasin) ---
# insightface modeli (buffalo_l) ~/.insightface altina iner -> HOME yazilabilir
# olmali. compose'taki kalici volume bu yola (HOME/.insightface) baglanir.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /home/appuser/.insightface \
    && chown -R appuser:appuser /app /home/appuser
ENV HOME=/home/appuser
USER appuser

# Flask/waitress web sunucusu portu (config.yaml: web.port)
EXPOSE 5000

# Tamponsuz log -> docker logs aninda goster
ENV PYTHONUNBUFFERED=1

# Saglik kontrolu: /healthz (kimlik gerektirmez). curl yok -> python ile.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/healthz', timeout=3).status==200 else 1)" || exit 1

# Headless yakalama + web arayuzu. insightface modeli ilk calismada
# ~/.insightface (=/home/appuser/.insightface) altina iner; compose'ta volume ile kalici.
CMD ["python", "webui.py"]
