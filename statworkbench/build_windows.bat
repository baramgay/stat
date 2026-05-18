@echo off
echo ==========================================
echo StatWorkbench Build
echo ==========================================
echo.

cd /d "%~dp0"

echo [*] Installing dependencies...
pip install --no-cache-dir PySide6 pandas numpy scipy statsmodels openpyxl pydantic tabulate pyinstaller 2>nul

echo [*] Installing StatWorkbench...
pip install --no-cache-dir -e . 2>nul

echo [*] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [*] Building executable...
python -m PyInstaller --name StatWorkbench --onefile --windowed --clean --noconfirm src\statworkbench\main.py

echo.
if exist dist\StatWorkbench.exe (
    echo [SUCCESS] Build complete!
    echo Location: %~dp0dist\StatWorkbench.exe
    dir dist\StatWorkbench.exe
) else (
    echo [ERROR] Build failed.
)

echo.
pause
