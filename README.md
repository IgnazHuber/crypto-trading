# Crypto Trading Projekt (Basis)

Dieses Projekt ist ein modularer Ansatz für die Analyse und das Backtesting von Kryptowährungen.  
Die Basis enthält:
- Datenabruf (yfinance, Binance-Dummy)
- Dummy-Handelsstrategie
- Einfachen Backtest
- Erste Visualisierung

---

## Voraussetzungen

- Python >= 3.10 (empfohlen)
- Git (für Versionsverwaltung)

---

## Installation (Windows)

```bash
# Projekt klonen (oder neues Verzeichnis anlegen)
git clone <repo-url> crypto_trading
cd crypto_trading

# Virtuelle Umgebung erstellen
python -m venv .venv

# Virtuelle Umgebung aktivieren
.venv\Scripts\activate   # (CMD)
# oder
.\.venv\Scripts\Activate.ps1   # (PowerShell)

# Abhängigkeiten installieren
pip install -r requirements.txt
