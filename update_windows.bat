@echo off
REM FaceZoom - Windows GUNCELLEME (CPU, cift tikla). PowerShell scriptini calistirir.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update_windows.ps1"
