# is_oos_blindtest_export_250805a.py
SKRIPT_ID = "is_oos_blindtest_export_250805a"
"""
IS/OOS-Optimierung & echter Blindtest.
Vollständiger Export: Alle Indikatorwerte/Parameter je Trade, Plotly-HTML, CSV/Excel.
"""

import os
import json
import pandas as pd
import inspect

from .data_loader_250805a import load_data
from .pattern_detection_250805a import PATTERN_FUNCTIONS
from .market_regime_250805a import classify_market_regime
from .multi_strategy_score_250805a import generate_score_signals
from .kpi_analyzer_250805a import calc_kpis

RESULTS_DIR = "results"
PARQUET_PATH = r"D:\Projekte\crypto_trading\crypto_trading\data\raw\BTCUSDT_1h_1year_ccxt.parquet"
START_CAPITAL = 10000
ASSET_NAME = "BTCUSDT"
# Du kannst nach Belieben weitere Parameter einfügen:
SESSION_PARAMS = {
    "atr_period": 14,
    "bb_window": 20,
    "min_signals_range": [1,2,3,4,5]
}

STRATEGY_CONFIG = [
    ("Many Trades Strategy", "always_trade", "trend", "Jede gerade Stunde (Demo)"),
    ("Bollinger Squeeze + Ausbruch Long", "bollinger_squeeze_long", "squeeze_breakout_long", "Squeeze + Breakout Long"),
    ("VWAP Breakout Long", "vwap_breakout_long", "uptrend", "Kurs > VWAP im Aufwärtstrend"),
    # ... ggf. weitere Strategien ...
]

def train_test_split_time(df, is_frac=0.6):
    split_idx = int(len(df) * is_frac)
    df_is = df.iloc[:split_idx].copy()
    df_oos = df.iloc[split_idx:].copy()
    return df_is, df_oos

def run_meta_strategy_with_indicators(
    df, signal_col, session_params, start_capital=10000, min_capital=100, asset="BTCUSDT", strategy_name="Meta_Score"
):
    trades = []
    capital = start_capital
    in_position = False
    entry_price, entry_idx, entry_time = None, None, None
    trade_id = 1
    for idx in df.index:
        if capital < min_capital:
            break
        entry_signal = df.loc[idx, signal_col]
        if not in_position and entry_signal:
            einsatz = min(0.10 * capital, capital)
            if einsatz < 1:
                continue
            in_position = True
            entry_price = df.loc[idx, 'close']
            entry_idx = idx
            entry_time = df.loc[idx, 'timestamp'] if 'timestamp' in df else idx
        elif in_position:
            price_now = df.loc[idx, 'close']
            stop_loss = entry_price * 0.95
            take_profit = entry_price * 1.20
            time_exceeded = (df.index.get_loc(idx) - df.index.get_loc(entry_idx) > 30)
            hit_stop = (price_now <= stop_loss)
            hit_tp = (price_now >= take_profit)
            if hit_stop or hit_tp or time_exceeded:
                exit_price = price_now
                exit_time = df.loc[idx, 'timestamp'] if 'timestamp' in df else idx
                pnl = (exit_price - entry_price) / entry_price * einsatz
                pnl = max(-0.05 * einsatz, min(pnl, 0.20 * einsatz))
                capital += pnl
                # Exportiere ALLE Indikatorwerte zum Entry!
                indics = {}
                for col in ['atr','bb_width','ema50','ema200','ema20','rsi','mfi','macd','vwap','ha_open','ha_close','bb_upper','bb_lower']:
                    indics[f"{col}_entry"] = float(df.loc[entry_idx, col]) if col in df else None
                trade_record = {
                    "Trade-ID": int(trade_id),
                    "Asset": asset,
                    "Strategy": strategy_name,
                    "Entry Time": entry_time,
                    "Exit Time": exit_time,
                    "Entry Price": round(entry_price, 2),
                    "Exit Price": round(exit_price, 2),
                    "Einsatz": round(einsatz, 2),
                    "PnL_abs": round(pnl, 2),
                    "PnL_pct": round(pnl / einsatz * 100, 2),
                    "Kapital nach Trade": round(capital, 2),
                }
                trade_record.update(indics)
                trade_record.update(session_params)
                trades.append(trade_record)
                trade_id += 1
                in_position = False
    trades_df = pd.DataFrame(trades)
    return trades_df

def call_plot_strategy_chart_with_equity(df, trades, strat_name, html_path):
    from .visualization_250805a import plot_strategy_chart_with_equity
    sig = inspect.signature(plot_strategy_chart_with_equity)
    params = list(sig.parameters.keys())
    try:
        if 'strat_name' in params:
            plot_strategy_chart_with_equity(df, trades, strat_name=strat_name, html_path=html_path)
        elif 'title' in params:
            plot_strategy_chart_with_equity(df, trades, title=strat_name, html_path=html_path)
        elif len(params) == 4:
            plot_strategy_chart_with_equity(df, trades, strat_name, html_path)
        elif len(params) == 3:
            plot_strategy_chart_with_equity(df, trades, html_path)
        else:
            raise TypeError(f"plot_strategy_chart_with_equity: Unbekannte Signatur: {params}")
    except Exception as e:
        print(f"[{SKRIPT_ID}] WARNUNG: Plot konnte nicht erzeugt werden: {e}")

def main():
    print(f"[{SKRIPT_ID}] Lade Daten und teile in IS/OOS ...")
    df = load_data(PARQUET_PATH)
    df_is, df_oos = train_test_split_time(df, is_frac=0.6)
    regimes_is = classify_market_regime(df_is)
    regimes_oos = classify_market_regime(df_oos)

    # --- 1. Optimierung NUR auf IS ---
    print(f"[{SKRIPT_ID}] Optimiere Score-Schwelle auf IS ...")
    grid_results = []
    for min_signals in SESSION_PARAMS["min_signals_range"]:
        score_df_is = generate_score_signals(df_is, regimes_is, STRATEGY_CONFIG, PATTERN_FUNCTIONS, min_signals=min_signals)
        trades_is = run_meta_strategy_with_indicators(
            df_is.assign(meta_long=score_df_is['meta_long']),
            'meta_long', {**SESSION_PARAMS, "min_signals": min_signals},
            start_capital=START_CAPITAL, asset=ASSET_NAME, strategy_name=f"Meta_Score_{min_signals}"
        )
        kpis_is = calc_kpis(trades_is, start_capital=START_CAPITAL)
        kpis_is['min_signals'] = min_signals
        kpis_is['trades'] = len(trades_is)
        grid_results.append(kpis_is)
    grid_df_is = pd.DataFrame(grid_results)
    grid_df_is.to_excel(os.path.join(RESULTS_DIR, "score_gridsearch_IS.xlsx"), index=False)
    best_row = grid_df_is.sort_values("Sharpe", ascending=False).iloc[0]
    best_min_signals = int(best_row['min_signals'])
    print(f"[{SKRIPT_ID}] Beste Score-Schwelle im IS: {best_min_signals}")

    # --- Parameter-Speicherung (JSON, für das Tradingmodul) ---
    global_params = {**SESSION_PARAMS, "min_signals": best_min_signals}
    with open(os.path.join(RESULTS_DIR, "best_params.json"), "w") as f:
        json.dump(global_params, f, indent=2)
    print(f"[{SKRIPT_ID}] Globale Parameter für Tradingmodul gespeichert: best_params.json")

    # --- 2. OOS-Blindtest ---
    print(f"[{SKRIPT_ID}] OOS-Blindtest mit IS-optimierten Parametern ...")
    score_df_oos = generate_score_signals(df_oos, regimes_oos, STRATEGY_CONFIG, PATTERN_FUNCTIONS, min_signals=best_min_signals)
    trades_oos = run_meta_strategy_with_indicators(
        df_oos.assign(meta_long=score_df_oos['meta_long']),
        'meta_long', global_params,
        start_capital=START_CAPITAL, asset=ASSET_NAME, strategy_name=f"Meta_Score_{best_min_signals}"
    )
    trades_oos.to_excel(os.path.join(RESULTS_DIR, "trades_OOS_full.xlsx"), index=False)
    trades_oos.to_csv(os.path.join(RESULTS_DIR, "trades_OOS_full.csv"), index=False)
    kpis_oos = calc_kpis(trades_oos, start_capital=START_CAPITAL)
    print(f"[{SKRIPT_ID}] OOS-KPIs:", kpis_oos)

    # --- 3. Interaktiver Plotly-HTML-Plot für OOS-Backtest mit Hoverinfo ---
    html_chart_path = os.path.join(RESULTS_DIR, f"OOS_Equity_Backtest.html")
    if not trades_oos.empty:
        call_plot_strategy_chart_with_equity(
            df_oos, trades_oos,
            strat_name=f"Meta_Score_OOS_{best_min_signals}",
            html_path=html_chart_path
        )
        print(f"[{SKRIPT_ID}] Interaktives OOS-Backtest-Chart: {html_chart_path}")
    else:
        print(f"[{SKRIPT_ID}] Keine Trades im OOS-Backtest, kein Chart erzeugt.")

    print(f"[{SKRIPT_ID}] IS/OOS-Test abgeschlossen. Alle relevanten Parameter und Trades exportiert.")

if __name__ == "__main__":
    main()
