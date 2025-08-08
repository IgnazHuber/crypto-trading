# d:\Projekte\crypto_trading\crypto_trading\visualization\compare_strategies_streamlit.py

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from crypto_trading.config.settings import RESULTS_DIR, PARQUET_PATH, safe_sheet_name

# --- Daten laden ---
EXCEL_PATH = os.path.join(RESULTS_DIR, "btc_candlestick_backtest_regime.xlsx")
df_all = pd.read_excel(EXCEL_PATH, sheet_name=None)
trades_dict = {k: v for k, v in df_all.items() if k not in ("KPIs", "Alle Trades")}

# Spaltennamen ggf. korrigieren
def standardize_columns(df):
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    rename_dict = {}
    for k in [
        "entrytime", "exittime", "entryprice", "exitprice",
        "tradeid", "einsatz", "pnl_abs", "pnlpct", "kapitalnachtrade"
    ]:
        for orig in df.columns:
            if orig.lower().replace(" ", "").replace("_", "") == k:
                rename_dict[orig] = {
                    "entrytime": "Entry Time",
                    "exittime": "Exit Time",
                    "entryprice": "Entry Price",
                    "exitprice": "Exit Price",
                    "tradeid": "trade_id",
                    "einsatz": "Einsatz",
                    "pnl_abs": "PnL_abs",
                    "pnlpct": "PnL_pct",
                    "kapitalnachtrade": "Kapital nach Trade"
                }[k]
    return df.rename(columns=rename_dict)

for k, trades in trades_dict.items():
    trades_dict[k] = standardize_columns(trades)
    # Zeitspalten umwandeln
    for c in ["Entry Time", "Exit Time"]:
        if c in trades_dict[k].columns and trades_dict[k][c].dtype == "object":
            trades_dict[k][c] = pd.to_datetime(trades_dict[k][c], errors="coerce")

# Lade Kursdaten (robust: Index & Spalten prüfen)
df = pd.read_parquet(PARQUET_PATH)

# 1. Falls Index ein Zeitindex ist → extrahiere als Spalte
if isinstance(df.index, pd.DatetimeIndex):
    df['timestamp'] = df.index

# 2. Falls noch keine Zeitspalte da: Erzeuge künstlich eine Zeitreihe
if 'timestamp' not in df.columns:
    start = pd.Timestamp("2023-01-01 00:00:00")   # Passe ggf. Start und Freq an!
    freq = "h"
    df['timestamp'] = pd.date_range(start, periods=len(df), freq=freq)

# 3. Sicherstellen, dass 'timestamp' vom richtigen Typ ist
if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# ---- Farben ----
COLORS = {
    1: {"win": "green", "loss": "red", "equity": "green"},
    2: {"win": "deepskyblue", "loss": "blue", "equity": "deepskyblue"}
}

def equity_curve(trades):
    if trades.empty:
        return [0], [0]
    trades_sorted = trades.sort_values('Exit Time')
    times = [trades_sorted.iloc[0]['Entry Time']]
    capitals = [trades_sorted.iloc[0]['Kapital nach Trade'] - trades_sorted.iloc[0]['PnL_abs']]
    for i, t in trades_sorted.iterrows():
        times.append(t['Exit Time'])
        capitals.append(t['Kapital nach Trade'])
    return times, [round(x, 1) for x in capitals]

def plot_compare(strat1, strat2):
    t1 = trades_dict[strat1]
    t2 = trades_dict[strat2]
    n1, n2 = len(t1), len(t2)
    p1, p2 = t1['PnL_abs'].sum() if n1 else 0, t2['PnL_abs'].sum() if n2 else 0
    pct1 = p1 / t1['Einsatz'].sum() * 100 if n1 else 0
    pct2 = p2 / t2['Einsatz'].sum() * 100 if n2 else 0

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            f"{strat1} Trades", 
            f"{strat2} Trades", 
            f"Beide Strategien: Trades im Kurs-Chart", 
            "Beide Strategien: Portfolioverlauf"
        ),
        vertical_spacing=0.08
    )

    # --- Einzelcharts oben ---
    for row, (s, t, c) in enumerate([(strat1, t1, COLORS[1]), (strat2, t2, COLORS[2])], 1):
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name="Kurs", showlegend=(row==1)
        ), row=row, col=1)
        for _, trade in t.iterrows():
            color = c["win"] if trade["PnL_abs"] >= 0 else c["loss"]
            fig.add_trace(go.Scatter(
                x=[trade['Entry Time']], y=[trade['Entry Price']],
                mode='markers+text',
                marker=dict(symbol='triangle-up', color=color, size=15, line=dict(width=2, color='black')),
                text=[f"{int(trade['trade_id'])}"], textfont=dict(size=10),
                textposition='top center',
                name=f"{s} Entry {int(trade['trade_id'])}", showlegend=False,
                hovertemplate=(
                    f"Trade-ID: {int(trade['trade_id'])}<br>"
                    f"Krypto: {trade['asset']}<br>"
                    f"Strategie: {trade['Strategy']}<br>"
                    f"Entry: {trade['Entry Time']}<br>"
                    f"Entry-Preis: {trade['Entry Price']:.1f}<br>"
                    f"Einsatz: {trade['Einsatz']:.1f}<br>"
                )
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=[trade['Exit Time']], y=[trade['Exit Price']],
                mode='markers+text',
                marker=dict(symbol='triangle-down', color=color, size=15, line=dict(width=2, color='black')),
                text=[f"{int(trade['trade_id'])}"], textfont=dict(size=10),
                textposition='bottom center',
                name=f"{s} Exit {int(trade['trade_id'])}", showlegend=False,
                hovertemplate=(
                    f"Trade-ID: {int(trade['trade_id'])}<br>"
                    f"Krypto: {trade['asset']}<br>"
                    f"Strategie: {trade['Strategy']}<br>"
                    f"Exit: {trade['Exit Time']}<br>"
                    f"Exit-Preis: {trade['Exit Price']:.1f}<br>"
                    f"Gewinn/Verlust: {trade['PnL_abs']:.1f} €<br>"
                    f"Gewinn/Verlust: {trade['PnL_pct']:.1f}%<br>"
                    f"Kapital nach Trade: {trade['Kapital nach Trade']:.1f} €<br>"
                )
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=[trade['Entry Time'], trade['Exit Time']],
                y=[trade['Entry Price'], trade['Exit Price']],
                mode='lines', line=dict(color=color, width=2, dash='dot'),
                showlegend=False
            ), row=row, col=1)

    # --- Gemeinsames Kurs-Chart, beide Strategien ---
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="Kurs"
    ), row=3, col=1)
    # Trades beider Strategien als Marker und Linien
    for t, c, name_prefix in [(t1, COLORS[1], strat1), (t2, COLORS[2], strat2)]:
        for _, trade in t.iterrows():
            color = c["win"] if trade["PnL_abs"] >= 0 else c["loss"]
            fig.add_trace(go.Scatter(
                x=[trade['Entry Time']], y=[trade['Entry Price']],
                mode='markers+text',
                marker=dict(symbol='triangle-up', color=color, size=12, line=dict(width=2, color='black')),
                text=[f"{int(trade['trade_id'])}"],
                textfont=dict(size=9),
                textposition='top center',
                name=f"{name_prefix} Entry {int(trade['trade_id'])}", showlegend=False,
                hovertemplate=(
                    f"Trade-ID: {int(trade['trade_id'])}<br>"
                    f"Krypto: {trade['asset']}<br>"
                    f"Strategie: {trade['Strategy']}<br>"
                    f"Entry: {trade['Entry Time']}<br>"
                    f"Entry-Preis: {trade['Entry Price']:.1f}<br>"
                    f"Einsatz: {trade['Einsatz']:.1f}<br>"
                )
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=[trade['Exit Time']], y=[trade['Exit Price']],
                mode='markers+text',
                marker=dict(symbol='triangle-down', color=color, size=12, line=dict(width=2, color='black')),
                text=[f"{int(trade['trade_id'])}"],
                textfont=dict(size=9),
                textposition='bottom center',
                name=f"{name_prefix} Exit {int(trade['trade_id'])}", showlegend=False,
                hovertemplate=(
                    f"Trade-ID: {int(trade['trade_id'])}<br>"
                    f"Krypto: {trade['asset']}<br>"
                    f"Strategie: {trade['Strategy']}<br>"
                    f"Exit: {trade['Exit Time']}<br>"
                    f"Exit-Preis: {trade['Exit Price']:.1f}<br>"
                    f"Gewinn/Verlust: {trade['PnL_abs']:.1f} €<br>"
                    f"Gewinn/Verlust: {trade['PnL_pct']:.1f}%<br>"
                    f"Kapital nach Trade: {trade['Kapital nach Trade']:.1f} €<br>"
                )
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=[trade['Entry Time'], trade['Exit Time']],
                y=[trade['Entry Price'], trade['Exit Price']],
                mode='lines', line=dict(color=color, width=1.5, dash='dot'),
                showlegend=False
            ), row=3, col=1)

    # --- Beide Equity-Kurven im gleichen Chart ---
    times1, capitals1 = equity_curve(t1)
    times2, capitals2 = equity_curve(t2)
    fig.add_trace(go.Scatter(
        x=times1, y=capitals1,
        mode='lines+markers', line=dict(color=COLORS[1]["equity"], width=3),
        marker=dict(size=5), name=f"{strat1} Equity"
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=times2, y=capitals2,
        mode='lines+markers', line=dict(color=COLORS[2]["equity"], width=3),
        marker=dict(size=5), name=f"{strat2} Equity"
    ), row=4, col=1)

    fig.update_layout(
        title={
            "text": (
                f"Vergleich: {strat1} (Trades: {n1}, G/V: {p1:.1f} €/{pct1:.1f} %) "
                f"vs. {strat2} (Trades: {n2}, G/V: {p2:.1f} €/{pct2:.1f} %)"
            ),
            "x": 0.5, "y": 0.97, "xanchor": "center", "yanchor": "top"
        },
        height=1800, template="plotly_white"
    )
    return fig

# ---- Streamlit App ----
st.set_page_config(layout="wide", page_title="Strategie-Vergleich (Streamlit)")
st.title("Krypto-Strategie-Vergleich")

sheetnames = list(trades_dict.keys())

col1, col2 = st.columns(2)
with col1:
    strat1 = st.selectbox("Strategie 1:", sheetnames, key="strat1")
with col2:
    strat2 = st.selectbox("Strategie 2:", sheetnames, index=1 if len(sheetnames) > 1 else 0, key="strat2")

st.plotly_chart(plot_compare(strat1, strat2), use_container_width=True)

# Optionale KPIs unter den Plot:
t1 = trades_dict[strat1]
t2 = trades_dict[strat2]
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**{strat1}**")
    st.write(f"Anzahl Trades: {len(t1)}")
    st.write(f"Gesamtgewinn/-verlust: {t1['PnL_abs'].sum():.1f} €")
    st.write(f"Durchschnittlicher Gewinn/Verlust: {t1['PnL_pct'].mean():.1f} %")
with col2:
    st.markdown(f"**{strat2}**")
    st.write(f"Anzahl Trades: {len(t2)}")
    st.write(f"Gesamtgewinn/-verlust: {t2['PnL_abs'].sum():.1f} €")
    st.write(f"Durchschnittlicher Gewinn/Verlust: {t2['PnL_pct'].mean():.1f} %")
