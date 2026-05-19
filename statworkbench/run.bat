@echo off
echo ==========================================
echo StatWorkbench
echo ==========================================
echo.

cd /d "%~dp0"

if exist dist\StatWorkbench\StatWorkbench.exe (
    echo [*] 실행 파일로 시작합니다...
    start dist\StatWorkbench\StatWorkbench.exe
    exit /b 0
)

echo [*] 실행 파일 없음. Python으로 실행합니다...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [*] Installing dependencies...
pip install -q --no-cache-dir PySide6 pandas numpy scipy statsmodels openpyxl pydantic tabulate 2>nul

echo [*] Installing StatWorkbench...
pip install -q --no-cache-dir -e . 2>nul

echo [*] Starting StatWorkbench...
echo.
set PYTHONPATH=src
python src\statworkbench\main.py

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start StatWorkbench.
    pause
)
