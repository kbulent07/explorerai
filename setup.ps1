# =============================================================================
# AiEye - Windows kurulum scripti (Docker) - HERSEY DAHIL
# -----------------------------------------------------------------------------
# Yaptiklari:
#   1) Docker Desktop var mi + motor calisiyor mu (degilse baslatir, bekler)
#   2) config.yaml yoksa sablondan olusturur
#   3) .env sifreleme anahtarini (AIEYE_SECRET_KEY) uretir (kamera parolalari
#      config'te sifreli tutulur)
#   4) YOLOX modelini indirir (giris/cikis sayimi icin, ~4 MB)
#   5) docker compose ile imaji derler + baslatir
#   6) Saglik durumunu bekler ve erisim adresini yazar
#
# CALISTIRMA:
#   - setup.bat'a CIFT TIKLA (onerilen), veya
#   - PowerShell'de:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
# =============================================================================

Set-Location -Path $PSScriptRoot
$ErrorActionPreference = 'Continue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Info($m){ Write-Host "[AiEye] $m" -ForegroundColor Cyan }
function Ok($m){   Write-Host "[TAMAM]  $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[UYARI]  $m" -ForegroundColor Yellow }
function Fail($m){ Write-Host "[HATA]   $m" -ForegroundColor Red }

Write-Host "==== AiEye Windows Kurulumu ====" -ForegroundColor White

if (-not (Test-Path "docker-compose.yml")) {
  Fail "docker-compose.yml bulunamadi. Bu scripti AiEye proje klasorunde calistirin."
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

# --- 1) Docker var mi? ---
Info "Docker kontrol ediliyor..."
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Fail "Docker bulunamadi. Once Docker Desktop kurun:"
  Write-Host "     https://www.docker.com/products/docker-desktop/" -ForegroundColor White
  Read-Host "Cikmak icin Enter"; exit 1
}

# --- 2) Docker motoru calisiyor mu? Degilse baslat + bekle ---
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

# --- 3) config.yaml ---
if (-not (Test-Path "config.yaml")) {
  Copy-Item "config.example.yaml" "config.yaml"
  Warn "config.yaml olusturuldu (sablondan)."
  Warn "-> Kameralari ekleyin ve web parolasini degistirin: config.yaml (veya arayuzden Kamera Ayarlari)."
} else {
  Ok "config.yaml mevcut."
}

# --- 4) .env sifreleme anahtari (BOM'suz UTF-8) ---
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

# --- 5) YOLOX modeli (giris/cikis sayimi icin) ---
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

# --- 6) Imaji derle + baslat ---
Info "Docker imaji derleniyor ve baslatiliyor (ilk sefer birkac dakika)..."
docker compose up -d --build
if ($LASTEXITCODE -ne 0) {
  Fail "docker compose build/up basarisiz oldu. Yukaridaki hatayi kontrol edin."
  Read-Host "Cikmak icin Enter"; exit 1
}

# --- 7) Saglik bekle ---
Info "Container sagligi bekleniyor..."
$healthy = $false
for ($i = 0; $i -lt 40; $i++) {
  Start-Sleep -Seconds 3
  $st = (docker inspect --format '{{.State.Health.Status}}' aieye 2>$null)
  if ($st -eq 'healthy') { $healthy = $true; break }
}
if ($healthy) { Ok "AiEye calisiyor (healthy)." }
else { Warn "Saglik dogrulanamadi. Loglara bakin:  docker compose logs -f" }

# --- 8) Erisim bilgisi ---
$ip = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
       Where-Object { $_.IPAddress -notmatch '^127\.' -and $_.IPAddress -notmatch '^169\.254' } |
       Select-Object -First 1).IPAddress
Write-Host ""
Write-Host "==== KURULUM TAMAM ====" -ForegroundColor White
Ok "Arayuz (bu PC):  http://localhost:5000"
if ($ip) { Ok "Agdaki cihazdan: http://$ip:5000" }
Info "Giris: config.yaml > web (varsayilan kullanici: admin)."
Info "Kameralar bu PC ile ayni yerel agda olmali (RTSP erisimi)."
Info "Durdurmak: docker compose down   |   Loglar: docker compose logs -f   |   Guncelle: docker compose up -d --build"
Write-Host ""
Read-Host "Kapatmak icin Enter"
