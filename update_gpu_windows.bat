@echo off
REM FaceZoom - Windows GUNCELLEME (GPU, cift tikla). PowerShell scriptini calistirir.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update_gpu_windows.ps1"
