@echo off
REM ============================================================
REM  start.bat — Launch Trading Lab (Bot + Dashboard)
REM  Run from the project root directory.
REM ============================================================

echo [Trading Lab] Starting trading bot...
start "Trading Bot" cmd /k ".venv\Scripts\python multi_symbol_loop.py"

REM Brief pause so the bot initialises before the dashboard loads
timeout /t 3 /nobreak > nul

echo [Trading Lab] Starting Streamlit dashboard...
start "Dashboard" cmd /k ".venv\Scripts\streamlit run dashboard.py"

echo.
echo Both processes launched in separate windows.
echo   Trading Bot  ^>  python multi_symbol_loop.py
echo   Dashboard    ^>  http://localhost:8501
echo.
pause
