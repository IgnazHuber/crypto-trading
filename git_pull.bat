@echo off
REM Holt aktuellen Stand von origin/main ins lokale Verzeichnis
REM Vor Benutzung: im Hauptprojektordner ausführen!

git status
git pull origin main

echo.
echo ==== Pull abgeschlossen! ====
pause
