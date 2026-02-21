@echo off
setlocal

set "ENTRY_DIR=%~dp0"
set "SCRIPT_PATH=%ENTRY_DIR%game_lifecycle_monitor.py"
set "RUNTIME_DIR=%ENTRY_DIR%runtime\logs"
set "JSONL_LOG=G:\Project\limbuscompany_Scripts3\runtime\logs\limbus_lifecycle.jsonl"
set "TEXT_LOG=G:\Project\limbuscompany_Scripts3\runtime\logs\limbus_lifecycle.log"

set "PYTHON_EXE=C:\Users\59863\anaconda3\envs\limbus\python.exe"
set "CONDA_BAT=C:\Users\59863\anaconda3\condabin\conda.bat"

if not exist "%SCRIPT_PATH%" (
  echo [ERROR] Script not found: %SCRIPT_PATH%
  pause
  exit /b 1
)

if not exist "%RUNTIME_DIR%" (
  mkdir "%RUNTIME_DIR%"
)

if exist "%PYTHON_EXE%" (
  echo [INFO] Using python: %PYTHON_EXE%
  start "Limbus Lifecycle Monitor" cmd /c ""%PYTHON_EXE%" "%SCRIPT_PATH%" --jsonl "%JSONL_LOG%" --text-log "%TEXT_LOG%" %*"
  exit /b 0
)

if exist "%CONDA_BAT%" (
  echo [INFO] Using conda env: limbus
  start "Limbus Lifecycle Monitor" cmd /c "call "%CONDA_BAT%" activate limbus && python "%SCRIPT_PATH%" --jsonl "%JSONL_LOG%" --text-log "%TEXT_LOG%" %*"
  exit /b 0
)

echo [ERROR] limbus conda environment not found.
echo [HINT] Expected:
echo        %PYTHON_EXE%
echo        or %CONDA_BAT%
pause
exit /b 1
