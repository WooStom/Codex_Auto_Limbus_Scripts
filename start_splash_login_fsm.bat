@echo off
setlocal

set "ENTRY_DIR=%~dp0"
set "SCRIPT_PATH=%ENTRY_DIR%splash_login_fsm.py"
set "PYTHON_EXE=C:\Users\59863\anaconda3\envs\limbus\python.exe"
set "CONDA_BAT=C:\Users\59863\anaconda3\condabin\conda.bat"

if not exist "%SCRIPT_PATH%" (
  echo [ERROR] Script not found: %SCRIPT_PATH%
  pause
  exit /b 1
)

if exist "%PYTHON_EXE%" (
  echo [INFO] Using python: %PYTHON_EXE%
  start "Limbus SPLASH LOGIN FSM" cmd /c ""%PYTHON_EXE%" "%SCRIPT_PATH%" %*"
  exit /b 0
)

if exist "%CONDA_BAT%" (
  echo [INFO] Using conda env: limbus
  start "Limbus SPLASH LOGIN FSM" cmd /c "call "%CONDA_BAT%" activate limbus && python "%SCRIPT_PATH%" %*"
  exit /b 0
)

echo [ERROR] limbus conda environment not found.
pause
exit /b 1
