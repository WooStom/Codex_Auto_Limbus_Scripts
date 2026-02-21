@echo off
setlocal

set "ENTRY_DIR=%~dp0"
set "RUNNER_SCRIPT=%ENTRY_DIR%run_monitor_and_fsm.py"
set "PYTHON_EXE=C:\Users\59863\anaconda3\envs\limbus\python.exe"
set "CONDA_BAT=C:\Users\59863\anaconda3\condabin\conda.bat"

if not exist "%RUNNER_SCRIPT%" (
  echo [ERROR] Not found: %RUNNER_SCRIPT%
  pause
  exit /b 1
)

if exist "%PYTHON_EXE%" (
  echo [INFO] Using python: %PYTHON_EXE%
  "%PYTHON_EXE%" "%RUNNER_SCRIPT%"
  exit /b %errorlevel%
)

if exist "%CONDA_BAT%" (
  echo [INFO] Using conda env: limbus
  call "%CONDA_BAT%" activate limbus
  python "%RUNNER_SCRIPT%"
  exit /b %errorlevel%
)

echo [ERROR] limbus conda environment not found.
pause
exit /b 1
