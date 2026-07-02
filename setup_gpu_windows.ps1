# =============================================================================
# AiEye - Windows GPU (NVIDIA/CUDA) kurulum scripti (Docker) - HERSEY DAHIL
# -----------------------------------------------------------------------------
# CPU'lu setup.ps1'ten FARKI:
#   - NVIDIA GPU + surucu var mi diye bakar (nvidia-smi)
#   - GPU imajini derler: docker-compose.gpu.yml override (Dockerfile.gpu,
#     onnxruntime-gpu + CUDA) -> YOLOX ve insightface CUDA'da kosar
#   - config.yaml'i GPU + YUKSEK DOGRULUK icin ayarlar:
#         compute_device: gpu           (CUDA saglayicisi)
#         cpu_profile: high             (tam coz. + her kare algilama)
#         recognition_det_size: 640     (yuz tanima dogrulugu ust seviye)
#
# GEREKSINIM (host):
#   - NVIDIA GPU + guncel surucu (nvidia-smi calisiyor olmali)
#   - Docker Desktop + NVIDIA Container Toolkit (GPU'yu container'a acar)
#     Docker Desktop (Windows) WSL2 backend ile GPU'yu otomatik destekler;
#     GPU'lu Linux host'ta NVIDIA Container Toolkit ayrica kurulmalidir.
#
# CALISTIRMA:
#   - setup_gpu_windows.bat'a CIFT TIKLA (onerilen), veya
#   - PowerShell'de:  powershell -ExecutionPolicy Bypass -File .\setup_gpu_windows.ps1
# =============================================================================

Set-Location -Path $PSScriptRoot
$ErrorActionPreference = 'Continue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Info($m){ Write-Host "[AiEye] $m" -ForegroundColor Cyan }
function Ok($m){   Write-Host "[TAMAM]  $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[UYARI]  $m" -ForegroundColor Yellow }
function Fail($m){ Write-Host "[HATA]   $m" -ForegroundColor Red }

# config.yaml icinde TOP-LEVEL (girintisiz) bir skaler anahtari ayarlar.
# Anahtar varsa degerini degistirir; yoksa dosya sonuna ekler. BOM'suz UTF-8 yazar
# (Python yaml yukleyicisi ile uyumlu olsun diye).
function Set-YamlScalar($file, $key, $value) {
  $lines = @(Get-Content -LiteralPath $file)
  $pattern = "^$([regex]::Escape($key))\s*:"
  $found = $false
  $out = foreach ($l in $lines) {
    if ($l -match $pattern) { $found = $true; "$key`: $value" } else { $l }
  }
  if (-not $found) { $out += "$key`: $value" }
  $enc = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllLines((Join-Path $PSScriptRoot $file), $out, $enc)
}

Write-Host "==== AiEye Windows GPU Kurulumu ====" -ForegroundColor White

if (-not (Test-Path "docker-compose.gpu.yml")) {
  Fail "docker-compose.gpu.yml bulunamadi. Bu scripti AiEye proje klasorunde calistirin."
  Read-Host "Cikmak icin Enter"; exit 1
}

# --- 0) Guncel kodu cek (git repo ise) ---
# config.yaml / .env .gitignore'da oldugundan pull bunlari EZMEZ.
Info "Guncel kod kontrol ediliyor (git pull)..."
if ((Get-Command git -ErrorAction SilentlyContinue) -and (Test-Path ".git")) {
  git pull --ff-only
  if ($LASTEXITCODE -eq 0) { Ok "Kod guncel." }
  else { Warn "git pull yapilamadi (yerel degisiklik/catisma veya internet yok). Mevcut kodla devam." }
} else {
  Info "Git deposu degil (veya git yok) -> kod guncellemesi atlandi; mevcut dosyalar kullanilir."
}

# --- 1) NVIDIA GPU / surucu var mi? ---
Info "NVIDIA GPU kontrol ediliyor (nvidia-smi)..."
$hasGpu = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  nvidia-smi | Out-Null
  if ($LASTEXITCODE -eq 0) { $hasGpu = $true }
}
if ($hasGpu) {
  Ok "NVIDIA GPU / surucu bulundu."
} else {
  Warn "nvidia-smi calismadi -> NVIDIA GPU/surucu bulunamadi."
  Warn "GPU imaji CPU-only makinede GEREKSIZ agirdir. GPU yoksa setup.bat (CPU) kullanin."
  $ans = Read-Host "Yine de GPU kurulumuna devam edilsin mi? (E/H)"
  if ($ans -notmatch '^[Ee]') { Info "Iptal edildi."; exit 0 }
}

# --- 2) Docker var mi? ---
Info "Docker kontrol ediliyor..."
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Fail "Docker bulunamadi. Once Docker Desktop kurun:"
  Write-Host "     https://www.docker.com/products/docker-desktop/" -ForegroundColor White
  Read-Host "Cikmak icin Enter"; exit 1
}

# --- 3) Docker motoru calisiyor mu? Degilse baslat + bekle ---
docker info 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
  Warn "Docker motoru kapali. Docker Desktop baslatiliyor (1-2 dk surebilir)..."
  $dd = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
  if (Test-Path $dd) { Start-Process $dd }
  $ready = $false
  for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 3
    docker info 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
  }
  if (-not $ready) {
    Fail "Docker motoru baslamadi. Docker Desktop'i acip 'Engine running' olunca tekrar calistirin."
    Read-Host "Cikmak icin Enter"; exit 1
  }
}
Ok "Docker hazir."

# --- 4) config.yaml ---
if (-not (Test-Path "config.yaml")) {
  Copy-Item "config.example.yaml" "config.yaml"
  Warn "config.yaml olusturuldu (sablondan)."
  Warn "-> Kameralari ekleyin ve web parolasini degistirin: config.yaml (veya arayuzden Kamera Ayarlari)."
} else {
  Ok "config.yaml mevcut."
}

# --- 5) config.yaml'i GPU + yuksek dogruluk icin ayarla ---
Info "config.yaml GPU + yuksek dogruluk icin ayarlaniyor..."
Set-YamlScalar "config.yaml" "compute_device" "gpu"
Set-YamlScalar "config.yaml" "cpu_profile" "high"
Set-YamlScalar "config.yaml" "recognition_det_size" "640"
Ok "compute_device: gpu | cpu_profile: high | recognition_det_size: 640"

# Yuksek dogruluk tanima modeli (antelopev2, ResNet100) - GPU'da onerilir.
# Ilk yuz gorulunce ~1 GB model iner (internet gerekir); istemeyen buffalo_l'de kalir.
$ansM = Read-Host "Yuksek dogruluk yuz tanima modeli (antelopev2, ilk seferde ~1 GB iner) kullanilsin mi? (E/H)"
if ($ansM -match '^[Ee]') {
  Set-YamlScalar "config.yaml" "recognition_model" "antelopev2"
  Ok "recognition_model: antelopev2 (yuksek dogruluk)"
} else {
  Set-YamlScalar "config.yaml" "recognition_model" "buffalo_l"
  Info "recognition_model: buffalo_l (standart). Sonra arayuzden antelopev2'ye gecebilirsiniz."
}

# --- 6) .env sifreleme anahtari (BOM'suz UTF-8) ---
$hasKey = (Test-Path ".env") -and (Select-String -Path ".env" -Pattern "AIEYE_SECRET_KEY=." -Quiet)
if (-not $hasKey) {
  $bytes = New-Object byte[] 32
  [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
  $key = [Convert]::ToBase64String($bytes).Replace('+','-').Replace('/','_')
  $enc = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::AppendAllText((Join-Path $PSScriptRoot ".env"), "AIEYE_SECRET_KEY=$key`n", $enc)
  Ok "Sifreleme anahtari uretildi (.env)."
  Warn "-> .env dosyasini KAYBETMEYIN; baska PC'ye tasirken birlikte goturun (yoksa parolalar cozulemez)."
} else {
  Ok "Sifreleme anahtari mevcut (.env)."
}

# --- 7) YOLOX modeli (giris/cikis sayimi icin) ---
if (-not (Test-Path "models")) { New-Item -ItemType Directory "models" | Out-Null }
if (-not (Test-Path "models\yolox_nano.onnx")) {
  Info "YOLOX modeli indiriliyor (~4 MB, giris/cikis sayimi icin)..."
  try {
    Invoke-WebRequest -UseBasicParsing `
      -Uri "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx" `
      -OutFile "models\yolox_nano.onnx"
    Ok "YOLOX modeli indirildi."
  } catch {
    Warn "YOLOX modeli indirilemedi (internet yok?). Sayim MediaPipe'a duser; modeli sonra models\ altina koyabilirsiniz."
  }
} else {
  Ok "YOLOX modeli mevcut."
}

# --- 8) GPU imajini derle + baslat (compose override) ---
Info "GPU Docker imaji derleniyor ve baslatiliyor (ilk sefer birkac dakika, CUDA tabani buyuk)..."
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
if ($LASTEXITCODE -ne 0) {
  Fail "GPU build/up basarisiz oldu. Yukaridaki hatayi kontrol edin."
  Warn "GPU container'a acilamadiysa: Docker Desktop > Settings > Resources > GPU acik mi?"
  Warn "GPU'lu Linux host'ta NVIDIA Container Toolkit kurulu mu? (nvidia-ctk)"
  Read-Host "Cikmak icin Enter"; exit 1
}

# --- 9) Saglik bekle ---
Info "Container sagligi bekleniyor..."
$healthy = $false
for ($i = 0; $i -lt 40; $i++) {
  Start-Sleep -Seconds 3
  $st = (docker inspect --format '{{.State.Health.Status}}' aieye 2>$null)
  if ($st -eq 'healthy') { $healthy = $true; break }
}
if ($healthy) { Ok "AiEye calisiyor (healthy, GPU)." }
else { Warn "Saglik dogrulanamadi. Loglara bakin:  docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f" }

# --- 10) GPU gercekten kullaniliyor mu? (log ipucu) ---
Info "GPU kullanimi loglardan dogrulaniyor..."
$logs = docker logs aieye 2>&1 | Select-String -Pattern "CUDA|GPU|CUDAExecutionProvider" | Select-Object -First 3
if ($logs) { $logs | ForEach-Object { Ok ("log: " + $_.ToString().Trim()) } }
else { Warn "Loglarda CUDA/GPU izi gorulmedi; ilk yuz gorulunce (insightface yuklenince) tekrar bakin." }

# --- 11) Erisim bilgisi ---
$ip = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
       Where-Object { $_.IPAddress -notmatch '^127\.' -and $_.IPAddress -notmatch '^169\.254' } |
       Select-Object -First 1).IPAddress
Write-Host ""
Write-Host "==== GPU KURULUM TAMAM ====" -ForegroundColor White
Ok "Arayuz (bu PC):  http://localhost:5000"
if ($ip) { Ok "Agdaki cihazdan: http://$ip:5000" }
Info "Ayarlar: compute_device=gpu, cpu_profile=high, recognition_det_size=640 (yuksek dogruluk)."
Info "Giris: config.yaml > web (varsayilan kullanici: admin)."
Info "Durdurmak: docker compose -f docker-compose.yml -f docker-compose.gpu.yml down"
Info "Loglar:    docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f"
Info "Guncelle:  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build"
Write-Host ""
Read-Host "Kapatmak icin Enter"
