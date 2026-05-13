@echo off
title OmniFace — ngrok HTTPS
cd /d "%~dp0"

echo Iniciando OmniFace con tunel HTTPS (ngrok)...
echo.

python run_ngrok.py %*

pause
