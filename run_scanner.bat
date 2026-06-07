@echo off
REM ============================================================
REM  Pullback Scanner — Windows Task Scheduler launcher
REM  Schedule this to run at 4:30 PM ET (Mon-Fri) or later
REM
REM  To schedule automatically:
REM    1. Open Task Scheduler (taskschd.msc)
REM    2. Create Basic Task > Daily > 4:30 PM
REM    3. Action: Start a Program
REM       Program: C:\Users\reese\frd_backtest\run_scanner.bat
REM    4. Set "Start in" to: C:\Users\reese\frd_backtest
REM ============================================================

cd /d "C:\Users\reese\frd_backtest"

REM Activate virtual environment if you use one (uncomment if needed)
REM call .venv\Scripts\activate.bat

python pullback_scanner.py >> scanner_results\scanner.log 2>&1

echo Scanner run complete: %date% %time% >> scanner_results\scanner.log
