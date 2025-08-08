@echo off
REM --- Pfade anpassen, falls nötig ---
set PROJECT_DIR=D:\Projekte\crypto_trading
set VENV_DIR=%PROJECT_DIR%\.venv

echo [INFO] Lösche Python-Cache...
for /d /r %PROJECT_DIR% %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

echo [INFO] Aktiviere virtuelles Environment...
call %VENV_DIR%\Scripts\activate.bat

echo [INFO] Starte Strategy Optimizer...
python -m crypto_trading.strategy_optimizer

pause
