# meta_optimizer_workflow_250805a.py
SKRIPT_ID = "meta_optimizer_workflow_250805a"
"""
Robuster Workflow: Score-Gridsearch, Parameter-Gridsearch, Portfolio-Backtest, Plot/Excel-Export.
Mit Debug-Checks & Simple-Strategy für viele Trades.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .data_loader_250805a import load_data
from .pattern_detection_250805a import PATTERN_FUNCTIONS
from .market_regime_250805a import classify_market_regime
from .multi_strategy_score_250805a import generate_score_signals
from .portfolio_backtester_250805a import run_meta_strategy
from .kpi_analyzer_250805a import calc_kpis

# --- User Config --- #
RESULTS_DIR = "results"
PARQUET_PATHS = {
    "BTCUSDT": r"D:\Projekte\crypto_trading\crypto_trading\data\raw\BTCUSDT_1h_1year_ccxt.parquet",
}
# Viele Trades: Simpel-Strategie (z.B. jede gerade Stunde ein Trade)
def always_trade(df):
    # Signal: jede gerade Stunde "True"
    return (df['timestamp'].dt.hour % 2 == 0)

# Temporär für diese Datei als Pattern-Funktion einbinden!
PATTERN_FUNCTIONS['always_trade'] = always_trade

STRATEGY_CONFIG = [
    ("Many Trades Strategy", "always_trade", "trend", "Jede gerade Stunde (Demo)"),
    # ... alle bisherigen Strategien ...
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
START_CAPITAL = 10000

os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_gridsearch(grid_df, xcol='min_signals', metrics=['Sharpe','CAGR','TotalPnL_pct']):
    if grid_df.empty or grid_df[metrics].sum().sum() == 0:
        print(f"[{SKRIPT_ID}] WARNUNG: Keine relevanten Daten für Plot ({xcol})!")
        return
    ax = grid_df.plot(x=xcol, y=metrics, marker='o', title=f"Performance vs. {xcol}")
    plt.grid(True); plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def export_results(df, filename):
    if df is None or df.empty:
        print(f"[{SKRIPT_ID}] WARNUNG: Kein Export – DataFrame ist leer!")
        return
    df.to_excel(filename, index=False)
    print(f"[{SKRIPT_ID}] Ergebnisse exportiert: {filename}")

def gridsearch_score(df, regimes, min_signals_range=[1,2,3,4,5], start_capital=10000):
    results = []
    for min_signals in min_signals_range:
        print(f"[{SKRIPT_ID}] Score-Gridsearch: min_signals={min_signals}")
        score_df = generate_score_signals(df, regimes, STRATEGY_CONFIG, PATTERN_FUNCTIONS, min_signals=min_signals)
        num_signals = score_df['meta_long'].sum()
        print(f"  Summe Score-Signale: {num_signals}")
        if num_signals == 0:
            print(f"[WARNUNG] Keine Meta-Signale für min_signals={min_signals}")
        trades = run_meta_strategy(df.assign(meta_long=score_df['meta_long']), 'meta_long', start_capital=start_capital)
        print(f"  Trades: {len(trades)}")
        kpis = calc_kpis(trades, start_capital=start_capital)
        kpis['min_signals'] = min_signals
        kpis['signals'] = num_signals
        results.append(kpis)
    return pd.DataFrame(results)

def gridsearch_parameters(parquet_path, atr_periods=[14, 21], bb_windows=[20, 30]):
    all_results = []
    for atr_p in atr_periods:
        for bb_win in bb_windows:
            print(f"[{SKRIPT_ID}] Parameter-Gridsearch: ATR={atr_p}, BB={bb_win}")
            df = load_data(parquet_path)
            df['atr'] = (df['high'] - df['low']).rolling(atr_p).mean()
            df['bb_upper'] = df['close'].rolling(bb_win).mean() + 2 * df['close'].rolling(bb_win).std()
            df['bb_lower'] = df['close'].rolling(bb_win).mean() - 2 * df['close'].rolling(bb_win).std()
            regimes = classify_market_regime(df)
            score_df = generate_score_signals(df, regimes, STRATEGY_CONFIG, PATTERN_FUNCTIONS, min_signals=2)
            num_signals = score_df['meta_long'].sum()
            print(f"  Summe Score-Signale: {num_signals}")
            trades = run_meta_strategy(df.assign(meta_long=score_df['meta_long']), 'meta_long', start_capital=START_CAPITAL)
            print(f"  Trades: {len(trades)}")
            kpis = calc_kpis(trades, start_capital=START_CAPITAL)
            kpis['atr_period'] = atr_p
            kpis['bb_window'] = bb_win
            kpis['signals'] = num_signals
            all_results.append(kpis)
    return pd.DataFrame(all_results)

def portfolio_backtest(asset_paths, start_capital=10000, min_signals=2):
    capitals = []
    all_kpis = []
    all_trades = []
    for asset, parquet_path in asset_paths.items():
        print(f"[{SKRIPT_ID}] Portfolio-Backtest für Asset: {asset}")
        df = load_data(parquet_path)
        regimes = classify_market_regime(df)
        score_df = generate_score_signals(df, regimes, STRATEGY_CONFIG, PATTERN_FUNCTIONS, min_signals=min_signals)
        num_signals = score_df['meta_long'].sum()
        print(f"  Summe Score-Signale: {num_signals}")
        trades = run_meta_strategy(df.assign(meta_long=score_df['meta_long']), 'meta_long', start_capital=start_capital, asset=asset)
        print(f"  Trades: {len(trades)}")
        if trades.empty:
            print(f"[WARNUNG] Keine Trades im Portfolio-Backtest für Asset: {asset}")
        kpis = calc_kpis(trades, start_capital=start_capital)
        capitals.append(trades['Kapital nach Trade'].values if not trades.empty else np.array([start_capital]))
        all_kpis.append({"Asset": asset, **kpis})
        trades['Asset'] = asset
        all_trades.append(trades)
    if not capitals:
        print(f"[WARNUNG] Portfolio-Backtest: Keine Kapitalkurven erzeugt.")
        portfolio_curve = np.zeros(1)
    else:
        maxlen = max(map(len,capitals))
        stack_curves = [c if len(c)==maxlen else np.pad(c,(0,maxlen-len(c)),'edge') for c in capitals]
        portfolio_curve = np.mean(np.stack(stack_curves), axis=0)
    return pd.DataFrame(all_kpis), pd.concat(all_trades, ignore_index=True), portfolio_curve

def main():
    print(f"[{SKRIPT_ID}] Starte robusten Meta-Optimizer-Workflow ...")
    # Score-Schwellen-Gridsearch (BTCUSDT)
    df = load_data(PARQUET_PATHS["BTCUSDT"])
    regimes = classify_market_regime(df)
    grid_df = gridsearch_score(df, regimes, min_signals_range=[1,2,3,4,5])
    export_results(grid_df, os.path.join(RESULTS_DIR, "score_gridsearch.xlsx"))
    plot_gridsearch(grid_df, xcol='min_signals', metrics=['Sharpe','CAGR','TotalPnL_pct'])
    # Einzelstrategie-Parameter-Gridsearch (ATR/BB)
    param_df = gridsearch_parameters(PARQUET_PATHS["BTCUSDT"], atr_periods=[14,21], bb_windows=[20,30])
    export_results(param_df, os.path.join(RESULTS_DIR, "parameter_gridsearch.xlsx"))
    plot_gridsearch(param_df, xcol='atr_period', metrics=['Sharpe','CAGR'])
    # Portfolio/Multi-Asset-Backtest
    kpis, trades, portfolio_curve = portfolio_backtest(PARQUET_PATHS, start_capital=START_CAPITAL, min_signals=2)
    export_results(kpis, os.path.join(RESULTS_DIR, "portfolio_kpis.xlsx"))
    export_results(trades, os.path.join(RESULTS_DIR, "portfolio_trades.xlsx"))
    if portfolio_curve is not None and len(portfolio_curve) > 1:
        plt.plot(portfolio_curve)
        plt.title("Portfolio-Kapitalkurve (Equal Weight)")
        plt.xlabel("Trades / Zeit")
        plt.ylabel("Kapital")
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    print(f"[{SKRIPT_ID}] Workflow abgeschlossen.")

if __name__ == "__main__":
    main()
