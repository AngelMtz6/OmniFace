@echo off
set PYTHON=C:\Users\angel\Desktop\Hackatec\OmniFace-1\.venv\Scripts\python.exe

echo ============================================
echo  OmniFace - GPU Diagnostic
echo ============================================
echo.

echo [1] onnxruntime instalado?
"%PYTHON%" -c "import onnxruntime; print('version:', onnxruntime.__version__)"
echo.

echo [2] Providers disponibles:
"%PYTHON%" -c "import onnxruntime; print(onnxruntime.get_available_providers())"
echo.

echo [3] Es onnxruntime-gpu o cpu?
"%PYTHON%" -c "import pkg_resources; p=pkg_resources.get_distribution('onnxruntime-gpu'); print('onnxruntime-gpu', p.version)" 2>nul || echo "onnxruntime-gpu NO instalado"
"%PYTHON%" -c "import pkg_resources; p=pkg_resources.get_distribution('onnxruntime'); print('onnxruntime (cpu)', p.version)" 2>nul || echo "onnxruntime-cpu no encontrado"
echo.

echo [4] CUDA disponible segun nvidia-smi?
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>nul || echo "nvidia-smi no encontrado en PATH"
echo.

echo [5] Version de CUDA toolkit:
nvcc --version 2>nul || echo "nvcc no en PATH (normal si no tienes CUDA toolkit instalado)"
echo.

pause
