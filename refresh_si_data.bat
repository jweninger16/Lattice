@echo off
REM Refresh short interest data - run weekly via Task Scheduler
cd /d C:\Users\jww9t\OneDrive\Desktop\swing_trader_v2\swing_trader_v2
call venv\Scripts\activate
python live/refresh_si_data.py >> logs\refresh_si.log 2>&1
