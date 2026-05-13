@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
"C:\Users\angel\Desktop\Hackatec\OmniFace-1\.venv\Scripts\pip.exe" install insightface > "C:\Users\angel\Desktop\Hackatec\OmniFace-1\install_log.txt" 2>&1
echo DONE > "C:\Users\angel\Desktop\Hackatec\OmniFace-1\install_done.txt"
