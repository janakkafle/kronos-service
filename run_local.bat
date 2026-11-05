@echo off
setlocal
cd /d "%~dp0.."
set PYTHONPATH=%CD%;%CD%\kronos;%PYTHONPATH%
echo ========================================================
echo Starting Kronos-base Financial API on http://127.0.0.1:8001
echo Working Directory: %CD%
echo ========================================================
python -m uvicorn kronos.app:app --host 127.0.0.1 --port 8001 --reload
pause
