@echo off
echo ==========================================
echo NuriStat Build
echo ==========================================
echo.

cd /d "%~dp0"

echo [*] Installing dependencies...
pip install --no-cache-dir PySide6 pandas numpy scipy statsmodels openpyxl pydantic tabulate pyinstaller 2>nul

echo [*] Installing NuriStat...
pip install --no-cache-dir -e . 2>nul

echo [*] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [*] Building executable...
python -m PyInstaller --name NuriStat --onefile --windowed --clean --noconfirm src\nuristat\main.py

echo.
if exist dist\NuriStat.exe (
    echo [SUCCESS] Build complete!
    echo Location: %~dp0dist\NuriStat.exe
    dir dist\NuriStat.exe
) else (
    echo [ERROR] Build failed.
)

echo.
pause
