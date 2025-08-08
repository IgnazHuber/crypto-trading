# btc-1h-1y-10kpi_250805a.py
SKRIPT_ID = "main_250805a"
"""
Komplettes Meta-Strategie- und Portfolio-Backtesting-Framework.
Mit Score-Bildung, Portfolio-Backtest, KPI-Auswertung, Gridsearch-Optimizer.
"""

import os
import pandas as pd
import time
from tqdm import tqdm

from .data_loader_250805a import SKRIPT_ID as DL_ID, load_data
from .pattern_detection_250805a import SKRIPT_ID as PD_ID, PATTERN_FUNCTIONS
from .market_regime_250805a import SKRIPT_ID as MR_ID, classify_market_regime

from .multi_strategy_score_250805a import SKRIPT_ID as MS_ID, generate_score_signals
from .portfolio_backtester_250805a import SKRIPT_ID as PB_ID, run_meta_strategy
from .kpi_analyzer_250805a import SKRIPT_ID as KPI_ID, calc_kpis
from .meta_gridsearch_250805a import SKRIPT_ID as OPT_ID, gridsearch_meta

# === Einstellungen ===
PARQUET_PATH = r"D:\Projekte\crypto_trading\crypto_trading\data\raw\BTCUSDT_1h_1year_ccxt.parquet"
RESULTS_DIR = "results"
START_CAPITAL = 10_000
ASSET = "BTCUSDT"

# STRATEGY_CONFIG enthält alle Pattern- und Score-basierten Strategien (siehe vorherige Tabellen)
STRATEGY_CONFIG = [
    ("Bollinger Squeeze + Ausbruch Long", "bollinger_squeeze_long", "squeeze_breakout_long", "Squeeze + Breakout Long"),
    ("Bollinger Squeeze + Ausbruch Short", "bollinger_squeeze_short", "squeeze_breakout_short", "Squeeze + Breakout Short"),
    ("VWAP Breakout Long", "vwap_breakout_long", "uptrend", "Kurs > VWAP im Aufwärtstrend"),
    ("VWAP Breakout Short", "vwap_breakout_short", "downtrend", "Kurs < VWAP im Abwärtstrend"),
    ("MFI Long (Oversold)", "mfi_long", "mfi_oversold", "MFI < 20 + Überverkauft"),
    ("MFI Short (Overbought)", "mfi_short", "mfi_overbought", "MFI > 80 + Überkauft"),
    ("Heikin Ashi Trend Long", "heikin_ashi_trend_long", "uptrend", "Heikin Ashi Trend + Aufwärtstrend"),
    ("Heikin Ashi Trend Short", "heikin_ashi_trend_short", "downtrend", "Heikin Ashi Trend + Abwärtstrend"),
    ("ATR Breakout Long", "atr_breakout_long", "high_volatility", "Kurs > High + ATR, bei hoher Volatilität"),
    ("ATR Breakout Short", "atr_breakout_short", "high_volatility", "Kurs < Low - ATR, bei hoher Volatilität"),
    ("Multi-TF EMA Crossover Long", "ema4h_crossover_long", "trend", "1h-Kurs > 4h-EMA + Trendfilter"),
    ("Multi-TF EMA Crossover Short", "ema4h_crossover_short", "downtrend", "1h-Kurs < 4h-EMA + Downtrend"),
]

os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print(f"\n[{SKRIPT_ID}] ================================")
    print(f"[{SKRIPT_ID}] Starte Meta-Strategie/Portfolio-Backtest für '{ASSET}'")
    start_time = time.time()

    # --- Schritt 1: Daten laden & Indikatoren berechnen ---
    print(f"[{SKRIPT_ID}] [{DL_ID}] Lade Kursdaten & berechne Indikatoren aus: {PARQUET_PATH}")
    df = load_data(PARQUET_PATH)
    print(f"[{SKRIPT_ID}] [{DL_ID}] Daten geladen: {len(df)} Zeilen\n")

    # --- Schritt 2: Marktregime erkennen ---
    print(f"[{SKRIPT_ID}] [{MR_ID}] Erkenne Marktregime ...")
    regimes = classify_market_regime(df)
    print(f"[{SKRIPT_ID}] [{MR_ID}] Marktregime-Spalten: {list(regimes.columns)}\n")

    # --- Schritt 3: Einzelstrategien - Signalscore generieren ---
    print(f"[{SKRIPT_ID}] [{MS_ID}] Generiere Score/Meta-Signale für alle Strategien ...")
    score_df = generate_score_signals(df, regimes, STRATEGY_CONFIG, PATTERN_FUNCTIONS, min_signals=2)
    print(score_df.head(10))
    print(f"[{SKRIPT_ID}] [{MS_ID}] Score/Meta-Signale generiert.\n")

    # --- Schritt 4: Meta-Backtest (Score, Multi-Strategie) ---
    print(f"[{SKRIPT_ID}] [{PB_ID}] Starte Portfolio-Backtest (Meta-Signal, Score >= 2) ...")
    meta_trades = run_meta_strategy(df.assign(meta_long=score_df['meta_long']), 'meta_long',
                                    start_capital=START_CAPITAL, asset=ASSET)
    print(f"[{SKRIPT_ID}] [{PB_ID}] Anzahl Trades: {len(meta_trades)}")

    # --- Schritt 5: KPI-Auswertung Meta-Strategie ---
    print(f"[{SKRIPT_ID}] [{KPI_ID}] KPI-Auswertung der Meta-Strategie ...")
    meta_kpis = calc_kpis(meta_trades, start_capital=START_CAPITAL)
    for k, v in meta_kpis.items():
        print(f"  {k}: {v}")
    print()

    # --- Schritt 6: Gridsearch/Optimizer: Optimiere Score-Schwelle ---
    print(f"[{SKRIPT_ID}] [{OPT_ID}] Starte Gridsearch/Optimizer über Score-Schwelle ...")
    grid_results = gridsearch_meta(df, regimes, STRATEGY_CONFIG, PATTERN_FUNCTIONS,
                                   min_signals_range=range(1, 7), start_capital=START_CAPITAL)
    grid_df = pd.DataFrame(grid_results)
    print(f"\n[{SKRIPT_ID}] Gridsearch-Optimizer Ergebnisse:")
    print(grid_df[['min_signals','Trades','Sharpe','TotalPnL_pct','MaxDrawdown','CAGR']])

    # Optional: Beste Schwelle/KPI speichern
    out_grid = os.path.join(RESULTS_DIR, "meta_strategy_gridsearch_results.xlsx")
    grid_df.to_excel(out_grid, index=False)
    print(f"\n[{SKRIPT_ID}] Gridsearch-Resultate gespeichert: {out_grid}")

    runtime = time.time() - start_time
    print(f"\n[{SKRIPT_ID}] Backtesting/Optimierung abgeschlossen. Laufzeit: {runtime:.1f} Sekunden.")
    print(f"[{SKRIPT_ID}] ================================\n")

if __name__ == "__main__":
    main()
