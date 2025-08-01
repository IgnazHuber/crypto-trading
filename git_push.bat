@echo off
REM Speichert alle aktuellen Änderungen, pusht zu origin/main
REM Vor Benutzung: im Hauptprojektordner ausführen!

git status
echo.
echo ==== STAGE ALL CHANGES ====
git add .
git status
echo.
set /p msg="Commit-Nachricht eingeben: "
git commit -m "%msg%"
git pull origin main
git push origin main

echo.
echo ==== Push abgeschlossen! ====
pause
