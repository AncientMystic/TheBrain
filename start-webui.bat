@echo off
REM Double-click to start TheBrain WebUI — no terminal skills needed.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe main.py --webui
) else (
  python main.py --webui
)
pause
