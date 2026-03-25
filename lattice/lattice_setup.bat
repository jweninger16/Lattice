@echo off
title Lattice Setup
echo.
echo  ============================================
echo   LATTICE — First Time Setup
echo  ============================================
echo.

cd /d "%~dp0.."
call venv\Scripts\activate.bat 2>nul

echo  Installing dependencies...
pip install fastapi uvicorn --quiet
echo  Done.
echo.

echo  Creating desktop shortcut...
powershell -ExecutionPolicy Bypass -File lattice\create_shortcut.ps1

echo.
echo  ============================================
echo   Setup complete!
echo   Double-click "Lattice" on your desktop.
echo  ============================================
echo.
pause
