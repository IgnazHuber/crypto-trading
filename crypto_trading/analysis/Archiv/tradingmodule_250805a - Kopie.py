# tradingmodule_250805a.py
SKRIPT_ID = "tradingmodule_250805a"
"""
Universelles Tradingmodul: Backtest auf beliebigem Parquet, Parameter aus externer Datei.
Zwei Charts: 1. Preis + Entry/Exit-Marker (mit Hover), 2. Equity-Kurve + Marker (mit Hover).
"""

import os
import json
import pandas as pd
import inspect
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from .data_loader_250805a import load_data
from .pattern_detection_250805a import PATTERN_FUNCTIONS
from .market_regime_250805a import classify_market_regime
from .multi_strategy_score_250805a import generate_score_signals

RESULTS_DIR = "results"
PARAMS_PATH = os.path.join(RESULTS_DIR, "best_params.json")
if not os.path.isfile(PARAMS_PATH):
    raise FileNotFoundError(f"Optimierte Parameter (best_params.json) nicht gefunden in {RESULTS_DIR}!")
with open(PARAMS_PATH, "r") as f:
    global_params = json.load(f)

PARQUET_PATH = input("Pfad zum Parquet-File: ").strip()
ASSET_NAME = input("Asset Name (z. B. BTCUSDT): ").strip()
START_CAPITAL = 10000

STRATEGY_CONFIG = [
    ("Many Trades Strategy", "always_trade", "trend", "Jede gerade Stunde (Demo)"),
    ("Bollinger Squeeze + Ausbruch Long", "bollinger_squeeze_long", "squeeze_breakout_long", "Squeeze + Breakout Long"),
    ("VWAP Breakout Long", "vwap_breakout_long", "uptrend", "Kurs > VWAP im Aufwärtstrend"),
    # ... weitere Strategien ...
]

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

def run_trading(df, regimes, params, asset_name):
    score_df = generate_score_signals(df, regimes, STRATEGY_CONFIG, PATTERN_FUNCTIONS, min_signals=params["min_signals"])
    trades = run_meta_strategy_with_indicators(
        df.assign(meta_long=score_df['meta_long']),
        'meta_long', params,
        start_capital=START_CAPITAL, asset=asset_name, strategy_name=f"Meta_Score_{params['min_signals']}"
    )
    return trades

def plot_dual_chart_with_markers(df, trades, html_path, strat_name="Meta_Score", asset=None):
    """
    Erstellt einen HTML-Plot mit 2 Subplots: oben Preis + Marker, unten Equity-Kurve + Marker.
    Alle Marker enthalten vollständige Hoverinfos.
    """
    if trades.empty:
        print("[plot_dual_chart_with_markers] Keine Trades, kein Chart erzeugt.")
        return
    if asset is None:
        asset = trades['Asset'].iloc[0] if 'Asset' in trades.columns else "Asset"
    # Subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Kursverlauf mit Trades", "Equity-Kurve mit Trades"),
                        vertical_spacing=0.08, row_heights=[0.6, 0.4])

    # Preis-Kurve oben
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['close'], mode="lines", name="Kurs", line=dict(color="gray")
    ), row=1, col=1)

    # Entry-Marker (Preis)
    entry_hover = [
        f"Trade-ID: {row['Trade-ID']}<br>"
        f"Asset: {row['Asset']}<br>"
        f"Strategy: {row['Strategy']}<br>"
        f"Entry: {row['Entry Time']}<br>"
        f"Entry-Preis: {row['Entry Price']}<br>"
        f"Einsatz: {row['Einsatz']}<br>"
        f"Indikatoren: ATR={row.get('atr_entry','')} | BB_Width={row.get('bb_width_entry','')}<br>"
        f"EMA50={row.get('ema50_entry','')}, RSI={row.get('rsi_entry','')}, MFI={row.get('mfi_entry','')}<br>"
        f"min_signals: {row.get('min_signals','')}, ATR_Period: {row.get('atr_period','')}, BB_Window: {row.get('bb_window','')}"
        for i, row in trades.iterrows()
    ]
    fig.add_trace(go.Scatter(
        x=trades["Entry Time"], y=trades["Entry Price"], mode="markers",
        marker=dict(color="green", symbol="triangle-up", size=12),
        name="Entry", hoverinfo="text", text=entry_hover
    ), row=1, col=1)

    # Exit-Marker (Preis)
    exit_hover = [
        f"Trade-ID: {row['Trade-ID']}<br>"
        f"Asset: {row['Asset']}<br>"
        f"Strategy: {row['Strategy']}<br>"
        f"Exit: {row['Exit Time']}<br>"
        f"Exit-Preis: {row['Exit Price']}<br>"
        f"PnL: {row['PnL_abs']} ({row['PnL_pct']}%)<br>"
        f"Equity: {row['Kapital nach Trade']}"
        for i, row in trades.iterrows()
    ]
    fig.add_trace(go.Scatter(
        x=trades["Exit Time"], y=trades["Exit Price"], mode="markers",
        marker=dict(color="red", symbol="triangle-down", size=12),
        name="Exit", hoverinfo="text", text=exit_hover
    ), row=1, col=1)

    # Equity-Kurve unten
    if "Kapital nach Trade" in trades.columns:
        fig.add_trace(go.Scatter(
            x=trades["Exit Time"], y=trades["Kapital nach Trade"],
            mode="lines+markers", name="Equity-Kurve", line=dict(color="blue", dash="dot")
        ), row=2, col=1)
        # Entry/Exit-Marker (Equity)
        fig.add_trace(go.Scatter(
            x=trades["Entry Time"], y=trades["Kapital nach Trade"], mode="markers",
            marker=dict(color="green", symbol="triangle-up", size=10),
            name="Entry (Equity)", hoverinfo="text", text=entry_hover, showlegend=False
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=trades["Exit Time"], y=trades["Kapital nach Trade"], mode="markers",
            marker=dict(color="red", symbol="triangle-down", size=10),
            name="Exit (Equity)", hoverinfo="text", text=exit_hover, showlegend=False
        ), row=2, col=1)
    fig.update_layout(
        title=f"{asset} Backtest: {strat_name}",
        xaxis_title="Zeit", yaxis_title="Preis",
        xaxis2_title="Zeit", yaxis2_title="Kapital",
        hovermode="closest",
        legend=dict(orientation="h"),
        height=850
    )
    fig.write_html(html_path)
    print(f"[plot_dual_chart_with_markers] Plotly-HTML exportiert: {html_path}")

def main():
    print(f"[{SKRIPT_ID}] Lade Parquet-Daten ...")
    df = load_data(PARQUET_PATH)
    regimes = classify_market_regime(df)
    trades = run_trading(df, regimes, global_params, ASSET_NAME)
    trades.to_excel(os.path.join(RESULTS_DIR, f"trades_{ASSET_NAME}.xlsx"), index=False)
    trades.to_csv(os.path.join(RESULTS_DIR, f"trades_{ASSET_NAME}.csv"), index=False)
    print(f"[{SKRIPT_ID}] Alle Trades für {ASSET_NAME} gespeichert.")

    html_chart_path = os.path.join(RESULTS_DIR, f"{ASSET_NAME}_Equity_Backtest.html")
    if not trades.empty:
        plot_dual_chart_with_markers(
            df, trades,
            html_chart_path, strat_name=f"Meta_Score_{global_params['min_signals']}_{ASSET_NAME}", asset=ASSET_NAME
        )
        print(f"[{SKRIPT_ID}] Interaktives Backtest-Chart: {html_chart_path}")
    else:
        print(f"[{SKRIPT_ID}] Keine Trades, kein Chart erzeugt.")

    print(f"[{SKRIPT_ID}] Tradingmodul-Backtest abgeschlossen.")

if __name__ == "__main__":
    main()
