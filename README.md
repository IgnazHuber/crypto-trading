# Projektzusammenfassung: Modulares Krypto-Analyse- und Backtesting-Framework

## Zielsetzung
- Entwicklung eines **modularen, skalierbaren Python-Frameworks** zur Analyse, Visualisierung und Bewertung von Kryptowährungen und Handelsstrategien.
- Fokus auf **automatisierte Datenbeschaffung**, technische Analyse, Backtesting und Reporting.
- Nahtlose Integration von **Krypto- und Indikator-Wissensdatenbanken** für interaktive Dashboards und Reports.

---

## Kernfunktionen

1. **Automatisierte Rohdatenbeschaffung**
    - Historische Preisdaten für alle wichtigen Kryptos (BTC, ETH, BNB, …) von Binance und anderen Quellen (ccxt, yfinance).
    - Unterstützung für verschiedene Zeitauflösungen (`1m`, `1h`, `1d`, …) und Zeitfenster (z. B. 1 Jahr / 1 Minute, 5 Jahre / 1 Tag).

2. **Krypto-Informationsdatenbank**
    - Automatische Erstellung einer **`crypto_infos.json`** mit CoinGecko-API:
        - Name, Symbol, Kurz- und Langinfo, Startdatum, Website, Marketcap, Preis.
    - Interaktive Konsolenanzeige und Selektion im Analyse-Workflow.

3. **Indikator-Informationsdatenbank**
    - **`indicator_db.py`** (bzw. als JSON):  
        - 100 wichtigste Trading-Indikatoren, sortiert nach Praxisrelevanz (MACD, RSI, EMA, …).
        - Für die Top 20 vollständige Bulletpoints (Kurzinfo, Marktumfeld, Werte, Kombinationen, Langinfo).
        - Platzhalter für alle weiteren, nachpflegbar.

4. **Interaktiver Info-Workflow**
    - Python-Modul für interaktive Selektion und Anzeige von Coins und Indikatoren (inkl. Detailansicht zu jedem Eintrag).
    - Konsolenmenü und Fortschrittsbalken.

5. **Chart- und Analyse-Tools**
    - Plotten von Candlestick-Charts, Portfolio-Übersichten und Radar-Charts (je Zeitfenster, Frequenz und Asset).
    - Vergleichende Analyse, Performance-Metriken, Reporting als PNG/PDF.

6. **Datenstruktur & Projektorganisation**
    - Trennung von Daten (`data/raw/`), Indikator-/Krypto-Infos (`indicator_db.py`, `crypto_infos.json`), Analyse/Visualisierung (`tools/`, `charts.py`), Tests (`tests/`).

---

## Highlights

- **Maximale Erweiterbarkeit:** Alle Listen und Datenbanken können iterativ ergänzt werden (z. B. weitere Coins, Indikatoren, Kennzahlen).
- **Automatisierung:** Datenfetch und Info-Generierung laufen automatisch, ohne manuelle Nachpflege.
- **Dokumentation und Menüführung:** Klar, robust und nachvollziehbar – geeignet für Wissenschaft, Lehre und Praxis.
- **Ideal für Forschung, Trading, Lehre, Reporting und Dashboards.**

---

## Typischer Workflow

1. **Rohdatenbeschaffung:**  
   Python-Skript lädt für alle gewünschten Coins/Intervalle die Daten, speichert als Parquet.

2. **Info-Generierung:**  
   Skript erzeugt/aktualisiert `crypto_infos.json` (Coingecko) und `indicator_db.py` (Indikatoren).

3. **Interaktive Analyse:**  
   Nutzer wählt Coin und Indikatoren, erhält sofort Kontext- und Detailinfos, kann Reports und Plots erzeugen.

4. **Visualisierung & Reporting:**  
   Candlestick-, Portfolio-, Radar-Charts, Performance-Auswertung und Export (PNG, PDF).

---

## Weiteres Vorgehen / ToDo

- Ausbau der Indikator-DB auf vollständige 100+ Bulletpoint-Sätze.
- Automatisierte Dashboards und PDF-Report-Module.
- Erweiterung um weitere Börsen und Datenquellen.
- Integration von Strategie-Backtests und Machine-Learning-Komponenten.

---

> Das Projekt ist so strukturiert, dass neue Module, Daten und Analysen jederzeit ergänzt werden können – alles bleibt versionierbar, dokumentiert und transparent.
