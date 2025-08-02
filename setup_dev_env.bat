@echo off
echo ================================
echo Entferne alte Pakete
echo ================================
pip freeze > old_requirements.txt
pip uninstall -y -r old_requirements.txt

echo ================================
echo Upgrade von pip
echo ================================
python -m pip install --upgrade pip

echo ================================
echo Installiere Anforderungen
echo ================================
pip install -r requirements.txt

echo ================================
echo Installiere Projekt als Dev
echo ================================
pip install -e .

echo ================================
echo Fertig! Teste Installation:
echo pytest -v --crypto=ss
echo ================================
