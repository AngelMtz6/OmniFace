@echo off
echo ============================================
echo  OmniFace - Reinstalling opencv-contrib
echo ============================================
echo.

set PIP=C:\Users\angel\Desktop\Hackatec\OmniFace-1\.venv\Scripts\pip.exe
set PYTHON=C:\Users\angel\Desktop\Hackatec\OmniFace-1\.venv\Scripts\python.exe

echo [1] Desinstalando cualquier resto de opencv...
"%PIP%" uninstall opencv-python opencv-python-headless opencv-contrib-python -y 2>nul
echo.

echo [2] Reinstalando opencv-contrib-python limpio...
"%PIP%" install opencv-contrib-python==4.10.0.84
echo.

echo [3] Verificando cv2...
"%PYTHON%" -c "import cv2; print('cv2 version:', cv2.__version__); r = cv2.face.LBPHFaceRecognizer_create(); print('cv2.face OK!')"
echo.

echo ============================================
echo  Done!
echo ============================================
pause
