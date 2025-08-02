# Crypto Trading Backtest Framework

**Vollautomatisierte Backtesting- und Strategie-Optimierung für Krypto-Portfolios (BTC, ETH, BNB, SOL, ...).  
Modular, robust, erweiterbar – inkl. PDF/Excel-Reporting und Indikator-Enzyklopädie.**

---

## 🚀 Projektübersicht

Dieses Python-Framework analysiert, optimiert und dokumentiert automatisiert Krypto-Trading-Strategien  
auf Basis von 30+ technischen Indikatoren.  
Ideal für professionelle Backtests, Strategieentwicklung, Portfolio-Analyse und Doku/Präsentation.

---

## ⚙️ Features

- **Multi-Asset-Backtesting:** Parallele Strategieprüfung auf BTC, ETH, BNB, SOL (u. v. m.).
- **Modulares Score-System:** Gewichtete Kombination aus 30+ Indikatoren (MACD, ADX, RSI, Volumen, BB, ...).
- **Grid-Search-Optimierung:** Systematische Suche nach optimalen Gewichtungen und Schwellen.
- **Automatischer Excel-Export:** Übersicht aller getesteten Strategiekombis mit Performance-Kennzahlen.
- **PDF-Reporting:** Mehrseitige Reports inkl. Trade-Tabelle, Scores, Portfolio-Kennzahlen, Indikator-Legende (Bulletpoints).
- **Indikatoren-Enzyklopädie:** Zentrale Doku für alle Indikatoren (Einsatz, Marktumfeld, Kombis, Standardwerte).
- **Fehlerrobust, testgetrieben:** Alle Regressionen abgedeckt; defensive Fehlerbehandlung in Daten und Berechnung.

---

## 📁 Verzeichnisstruktur

```plaintext
d:\Projekte\crypto_trading\
│
├── crypto_trading/
│   ├── indicators.py           # Zentrale Indikator-Berechnung (30+ Indikatoren, robust)
│   ├── indicator_legend.py     # Indikator-Legende (Bulletpoints, Marktumfeld, Kombis, Standardwerte)
│   ├── trades.py               # Trade-Engine, Score-System, DataFrame-Ausgabe
│   ├── strategy_optimizer.py   # Grid-Search, Top-N-Auswertung, Excel/PDF-Export
│   ├── data/
│   │   └── raw/                # Parquet-Daten (BTC_USD, ETH_USD, ...)
│   ├── visualization/
│   │   └── pdf_report.py       # PDF-Export inkl. Tabellen und Indikator-Legende
│   └── ...
│
├── results/
│   ├── strategy_grid_search.xlsx    # Alle getesteten Gewichtungs-Kombis und Kennzahlen
│   ├── strategy_top_1.pdf           # PDF-Report Top-Strategie (mit Indikator-Legende)
│   ├── strategy_top_2.pdf
│   └── ...
│
├── requirements.txt
└── README.md (diese Datei)
