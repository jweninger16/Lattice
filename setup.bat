@echo off
echo ============================================
echo   Swing Trader - Environment Setup (Windows)
echo ============================================
echo.

REM Try to find Python 3.11 specifically
set PYTHON_CMD=

REM Check py launcher first (most reliable on Windows)
py -3.11 --version >NUL 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.11
    goto :found
)

REM Check direct python3.11 command
python3.11 --version >NUL 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3.11
    goto :found
)

REM Check common install paths
if exist "C:\Python311\python.exe" (
    set PYTHON_CMD=C:\Python311\python.exe
    goto :found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
    goto :found
)

echo ERROR: Python 3.11 not found.
echo Please install it from: https://www.python.org/downloads/release/python-3119/
echo Make sure to check "Add Python to PATH" during installation.
pause
exit /b 1

:found
echo Found Python 3.11: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

echo [1/4] Creating virtual environment with Python 3.11...
%PYTHON_CMD% -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [4/4] Installing dependencies...
pip install -r requirements.txt

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo To activate your environment in future sessions:
echo   venv\Scripts\activate
echo.
echo To run the system:
echo   python main.py pipeline      ^<-- Download data (run first)
echo   python main.py signals       ^<-- See today's signals
echo   python main.py backtest      ^<-- Run backtest
echo   python main.py walkforward   ^<-- Walk-forward validation
echo.
pause
