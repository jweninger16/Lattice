@echo off
cd /d "%~dp0.."
call venv\Scripts\activate.bat 2>nul
start /min "" python lattice\launch.py
