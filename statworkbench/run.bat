@echo off
echo ==========================================
echo StatWorkbench
echo ==========================================
echo.

cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo [*] Installing dependencies...
pip install -q --no-cache-dir PySide6 pandas numpy scipy statsmodels openpyxl pydantic tabulate 2>nul

echo [*] Installing StatWorkbench...
pip install -q --no-cache-dir -e . 2>nul

echo [*] Starting StatWorkbench...
echo.
python -m statworkbench.main

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start StatWorkbench.
    pause
)
