@echo off
REM FaceZoom - Windows kurulum (cift tikla). PowerShell scriptini calistirir.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
