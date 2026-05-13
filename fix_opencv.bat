@echo off
echo ============================================
echo  OmniFace - Fixing OpenCV conflict
echo ============================================
echo.

echo [1] Uninstalling opencv-python-headless...
"C:\Users\angel\Desktop\Hackatec\OmniFace-1\.venv\Scripts\pip.exe" uninstall opencv-python-headless -y
echo.

echo [2] Verifying opencv-contrib-python is still installed...
"C:\Users\angel\Desktop\Hackatec\OmniFace-1\.venv\Scripts\pip.exe" show opencv-contrib-python
echo.

echo [3] Checking cv2.face module...
"C:\Users\angel\Desktop\Hackatec\OmniFace-1\.venv\Scripts\python.exe" -c "import cv2; print('cv2 version:', cv2.__version__); r = cv2.face.LBPHFaceRecognizer_create(); print('cv2.face OK - LBPH works!')"
echo.

echo ============================================
echo  Done! You can now run the app normally.
echo ============================================
pause
