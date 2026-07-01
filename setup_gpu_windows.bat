@echo off
REM FaceZoom - Windows GPU (NVIDIA/CUDA) kurulum (cift tikla). PowerShell scriptini calistirir.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_gpu_windows.ps1"
