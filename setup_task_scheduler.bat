@echo off
echo ============================================
echo   Swing Trader — Task Scheduler Setup
echo ============================================
echo.
echo This will create two Windows scheduled tasks:
echo   1. Morning Briefing  — 8:30 AM CST, Mon-Fri
echo   2. Intraday Monitor  — 8:35 AM CST, Mon-Fri
echo.
echo The monitor runs continuously until market close (3:00 PM CST).
echo.

REM Get the directory where this batch file lives
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

REM Build paths
set PYTHON=%SCRIPT_DIR%\venv\Scripts\python.exe
set MAIN=%SCRIPT_DIR%\main.py

REM Create logs directory
mkdir "%SCRIPT_DIR%\logs" 2>nul

REM ── Task 1: Morning Briefing ──────────────────────────────────────

set DAILY_WRAPPER=%SCRIPT_DIR%\run_daily.bat
echo @echo off > "%DAILY_WRAPPER%"
echo cd /d "%SCRIPT_DIR%" >> "%DAILY_WRAPPER%"
echo echo [%%date%% %%time%%] Starting morning briefing... >> "%SCRIPT_DIR%\logs\daily.log" >> "%DAILY_WRAPPER%"
echo "%PYTHON%" "%MAIN%" daily >> "%SCRIPT_DIR%\logs\daily.log" 2^>^&1 >> "%DAILY_WRAPPER%"

REM Delete existing task if it exists
schtasks /delete /tn "SwingTraderDaily" /f >nul 2>&1

REM Create: 8:30 AM Mon-Fri
schtasks /create ^
  /tn "SwingTraderDaily" ^
  /tr "\"%DAILY_WRAPPER%\"" ^
  /sc weekly ^
  /d MON,TUE,WED,THU,FRI ^
  /st 08:30 ^
  /rl highest ^
  /f ^
  /ru "%USERNAME%"

echo.
if %ERRORLEVEL% EQU 0 (
    echo [OK] Morning briefing task created — 8:30 AM CST, Mon-Fri
) else (
    echo [FAIL] Morning briefing task — try running as Administrator
)

REM ── Task 2: Intraday Monitor ──────────────────────────────────────

set MONITOR_WRAPPER=%SCRIPT_DIR%\run_monitor.bat
echo @echo off > "%MONITOR_WRAPPER%"
echo cd /d "%SCRIPT_DIR%" >> "%MONITOR_WRAPPER%"
echo echo [%%date%% %%time%%] Starting intraday monitor... >> "%SCRIPT_DIR%\logs\monitor.log" >> "%MONITOR_WRAPPER%"
echo "%PYTHON%" "%MAIN%" monitor >> "%SCRIPT_DIR%\logs\monitor.log" 2^>^&1 >> "%MONITOR_WRAPPER%"

REM Delete existing task if it exists
schtasks /delete /tn "SwingTraderMonitor" /f >nul 2>&1

REM Create: 8:35 AM Mon-Fri, auto-stop after 8 hours (covers market close)
schtasks /create ^
  /tn "SwingTraderMonitor" ^
  /tr "\"%MONITOR_WRAPPER%\"" ^
  /sc weekly ^
  /d MON,TUE,WED,THU,FRI ^
  /st 08:35 ^
  /rl highest ^
  /f ^
  /ru "%USERNAME%" ^
  /du 08:00

echo.
if %ERRORLEVEL% EQU 0 (
    echo [OK] Intraday monitor task created — 8:35 AM CST, Mon-Fri (8hr duration)
) else (
    echo [FAIL] Intraday monitor task — try running as Administrator
)

echo.
echo ============================================
echo   Summary
echo ============================================
echo.
echo   Morning Briefing:  8:30 AM CST, Mon-Fri
echo     - Downloads fresh data
echo     - Scores with ML model
echo     - Texts you buy/sell actions
echo.
echo   Intraday Monitor:  8:35 AM - 4:35 PM CST, Mon-Fri
echo     - Checks positions every 30 min
echo     - Texts you if stop/target hit
echo     - Auto-stops after 8 hours
echo.
echo   Logs saved to: %SCRIPT_DIR%\logs\
echo.
echo   To test now:
echo     schtasks /run /tn "SwingTraderDaily"
echo     schtasks /run /tn "SwingTraderMonitor"
echo.
echo   To remove:
echo     schtasks /delete /tn "SwingTraderDaily" /f
echo     schtasks /delete /tn "SwingTraderMonitor" /f
echo.
pause
