# =============================================================================
# AiEye - Windows GUNCELLEME scripti (GPU / NVIDIA-CUDA)
# -----------------------------------------------------------------------------
# GPU kurulumu YAPILMIS varsayar: sadece EN GUNCEL kodu ceker, GPU imajini
# yeniden derler ve baslatir. config.yaml / .env / modeller / veriler KORUNUR.
#   - Ilk GPU kurulumu icin:  setup_gpu_windows.bat  (config=gpu/high, model dahil)
#   - Bu script:              git pull + GPU compose override ile up -d --build
#
# CALISTIRMA:
#   - update_gpu_windows.bat'a CIFT TIKLA (onerilen), veya
#   - PowerShell'de:  powershell -ExecutionPolicy Bypass -File .\update_gpu_windows.ps1
# =============================================================================

Set-Location -Path $PSScriptRoot
$ErrorActionPreference = 'Continue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Info($m){ Write-Host "[AiEye] $m" -ForegroundColor Cyan }
function Ok($m){   Write-Host "[TAMAM]  $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[UYARI]  $m" -ForegroundColor Yellow }
function Fail($m){ Write-Host "[HATA]   $m" -ForegroundColor Red }

Write-Host "==== AiEye Windows Guncelleme (GPU) ====" -ForegroundColor White

if (-not (Test-Path "docker-compose.gpu.yml")) {
  Fail "docker-compose.gpu.yml bulunamadi. Bu scripti AiEye proje klasorunde calistirin."
  Read-Host "Cikmak icin Enter"; exit 1
}

# --- 1) Docker motoru calisiyor mu? ---
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Fail "Docker bulunamadi. Once Docker Desktop kurun (bkz. setup)."
  Read-Host "Cikmak icin Enter"; exit 1
}
docker info 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
  Warn "Docker motoru kapali. Docker Desktop baslatiliyor..."
  $dd = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
  if (Test-Path $dd) { Start-Process $dd }
  $ready = $false
  for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 3
    docker info 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
  }
  if (-not $ready) { Fail "Docker motoru baslamadi."; Read-Host "Cikmak icin Enter"; exit 1 }
}
Ok "Docker hazir."

# --- 2) Guncel kodu cek (git repo ise) ---
Info "Guncel kod cekiliyor (git pull)..."
if ((Get-Command git -ErrorAction SilentlyContinue) -and (Test-Path ".git")) {
  git pull --ff-only
  if ($LASTEXITCODE -eq 0) { Ok "Kod guncel." }
  else { Warn "git pull yapilamadi (yerel degisiklik/catisma veya internet yok). Mevcut kodla devam." }
} else {
  Warn "Git deposu degil (veya git yok) -> kod cekilemedi; mevcut dosyalarla yeniden derlenir."
}

# --- 3) GPU imajini yeniden derle + baslat (compose override) ---
Info "GPU imaji yeniden derleniyor ve baslatiliyor..."
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
if ($LASTEXITCODE -ne 0) {
  Fail "GPU build/up basarisiz. Yukaridaki hatayi kontrol edin."
  Warn "GPU acilamadiysa: Docker Desktop > Settings > Resources > GPU acik mi? / NVIDIA Container Toolkit?"
  Read-Host "Cikmak icin Enter"; exit 1
}

# --- 4) Saglik bekle ---
Info "Container sagligi bekleniyor..."
$healthy = $false
for ($i = 0; $i -lt 40; $i++) {
  Start-Sleep -Seconds 3
  $st = (docker inspect --format '{{.State.Health.Status}}' aieye 2>$null)
  if ($st -eq 'healthy') { $healthy = $true; break }
}
if ($healthy) { Ok "AiEye guncellendi ve calisiyor (healthy, GPU)." }
else { Warn "Saglik dogrulanamadi. Loglar: docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f" }

# --- 5) GPU kullanimi log ipucu ---
$logs = docker logs aieye 2>&1 | Select-String -Pattern "CUDA|GPU|CUDAExecutionProvider" | Select-Object -First 3
if ($logs) { $logs | ForEach-Object { Ok ("log: " + $_.ToString().Trim()) } }

# --- 6) Erisim bilgisi ---
$ip = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
       Where-Object { $_.IPAddress -notmatch '^127\.' -and $_.IPAddress -notmatch '^169\.254' } |
       Select-Object -First 1).IPAddress
Write-Host ""
Write-Host "==== GUNCELLEME TAMAM (GPU) ====" -ForegroundColor White
Ok "Arayuz (bu PC):  http://localhost:5000"
if ($ip) { Ok "Agdaki cihazdan: http://$ip:5000" }
Info "Loglar:  docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f"
Info "Durdur:  docker compose -f docker-compose.yml -f docker-compose.gpu.yml down"
Write-Host ""
Read-Host "Kapatmak icin Enter"
