@echo off
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
python portfolio.py --interactive --rf 0.04
if errorlevel 1 (
  echo.
  echo [!] Failed. Install deps first:
  echo     pip install pandas yfinance quantstats PyPortfolioOpt
)
echo.
pause
