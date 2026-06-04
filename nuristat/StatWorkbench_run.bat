@echo off
echo ==========================================
echo NuriStat Execution
echo ==========================================
echo.

cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo [Error] Python is not installed.
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip install -q PySide6 pandas numpy scipy statsmodels openpyxl pydantic tabulate

echo [2/3] Installing NuriStat...
pip install -q -e .

echo [3/3] Launching NuriStat...
echo.
python -m nuristat.main

if errorlevel 1 (
    echo.
    echo [Error] An error occurred during execution.
    pause
)
