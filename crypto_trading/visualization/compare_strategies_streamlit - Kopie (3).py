import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from crypto_trading.config.settings import RESULTS_DIR, PARQUET_PATH, safe_sheet_name

EXCEL_PATH = os.path.join(RESULTS_DIR, "btc_candlestick_backtest_regime.xlsx")
df_all = pd.read_excel(EXCEL_PATH, sheet_name=None)
trades_dict = {k: v for k, v in df_all.items() if k not in ("KPIs", "Alle Trades")}

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
    for c in ["Entry Time", "Exit Time"]:
        if c in trades_dict[k].columns and trades_dict[k][c].dtype == "object":
            trades_dict[k][c] = pd.to_datetime(trades_dict[k][c], errors="coerce")

df = pd.read_parquet(PARQUET_PATH)

# --- Zeitspalte erzeugen, falls nicht vorhanden ---
if 'timestamp' not in df.columns:
    if isinstance(df.index, pd.DatetimeIndex):
        df['timestamp'] = df.index
    else:
        # Echten frühesten Zeitpunkt aus ALLEN Trades extrahieren
        trade_times = []
        for trades in trades_dict.values():
            try:
                trade_times.append(pd.to_datetime(trades['Entry Time']).min())
            except Exception:
                continue
        if trade_times:
            start = min(trade_times)
        else:
            start = pd.Timestamp("2024-01-01 00:00:00")
        freq = "h"
        df['timestamp'] = pd.date_range(start, periods=len(df), freq=freq)
if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# --- Stimmt Zeitachse? Prüfen ---
st.write("Erster Timestamp Kursdaten:", df['timestamp'].min())
st.write("Letzter Timestamp Kursdaten:", df['timestamp'].max())

def equity_curve(trades):
    if trades.empty:
        return [0], [0]
    trades_sorted = trades.sort_values('Exit Time')
    times = [trades_sorted.iloc[0]['Entry Time']]
    capitals = [trades_sorted.iloc[0]['Kapital nach Trade'] - trades_sorted.iloc[0]['PnL_abs']]
    for _, t in trades_sorted.iterrows():
        times.append(t['Exit Time'])
        capitals.append(t['Kapital nach Trade'])
    return times, [round(x, 1) for x in capitals]

# Farben für Kombi-Chart (schwarz/orange für Marker)
COLORS_CANDLE = {1: {"win": "green", "loss": "red"}, 2: {"win": "deepskyblue", "loss": "blue"}}
MARKER_COLORS = {1: "black", 2: "orange"}

st.set_page_config(layout="wide", page_title="Strategie-Vergleich (Streamlit)")
st.title("Krypto-Strategie-Vergleich")

sheetnames = list(trades_dict.keys())

col1, col2 = st.columns(2)
with col1:
    strat1 = st.selectbox("Strategie 1:", sheetnames, key="strat1")
with col2:
    strat2 = st.selectbox("Strategie 2:", sheetnames, index=1 if len(sheetnames) > 1 else 0, key="strat2")

t1 = trades_dict[strat1]
t2 = trades_dict[strat2]

# --- Timestamps angleichen: Snap auf nächstgelegene Candle-Zeit ---
def align_to_candle(t, df):
    candle_times = pd.Series(df['timestamp'].unique())
    t = t.copy()
    t['Entry Time'] = pd.to_datetime(t['Entry Time'])
    t['Exit Time'] = pd.to_datetime(t['Exit Time'])
    t['Entry Time'] = t['Entry Time'].apply(lambda x: candle_times.iloc[(candle_times - x).abs().argmin()])
    t['Exit Time'] = t['Exit Time'].apply(lambda x: candle_times.iloc[(candle_times - x).abs().argmin()])
    return t

t1 = align_to_candle(t1, df)
t2 = align_to_candle(t2, df)

# --- Kurzanalyse für beide Strategien direkt unter der Auswahl ---
def kpi_block(name, trades):
    st.markdown(f"**{name}**")
    st.write(f"Anzahl Trades: {len(trades)}")
    st.write(f"Gesamtgewinn/-verlust: {trades['PnL_abs'].sum():.1f} €")
    st.write(f"Durchschnittlicher Gewinn/Verlust: {trades['PnL_pct'].mean():.1f} %")
k1, k2 = st.columns(2)
with k1:
    kpi_block(strat1, t1)
with k2:
    kpi_block(strat2, t2)

# --- Zwei Candlestick-Charts nebeneinander ---
def plot_single_candle(df, trades, strat, colorset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="Kurs"
    ))
    for _, trade in trades.iterrows():
        color = colorset["win"] if trade["PnL_abs"] >= 0 else colorset["loss"]
        fig.add_trace(go.Scatter(
            x=[trade['Entry Time']], y=[trade['Entry Price']],
            mode='markers+text',
            marker=dict(symbol='triangle-up', color=color, size=13, line=dict(width=2, color='black')),
            text=[f"{int(trade['trade_id'])}"], textfont=dict(size=10, color='black'),
            textposition='top center',
            name=f"Entry {int(trade['trade_id'])}", showlegend=False,
            hovertemplate=(
                f"Trade-ID: {int(trade['trade_id'])}<br>"
                f"Entry: {trade['Entry Time']}<br>"
                f"Entry-Preis: {trade['Entry Price']:.1f}<br>"
                f"Einsatz: {trade['Einsatz']:.1f}<br>"
            )
        ))
        fig.add_trace(go.Scatter(
            x=[trade['Exit Time']], y=[trade['Exit Price']],
            mode='markers+text',
            marker=dict(symbol='triangle-down', color=color, size=13, line=dict(width=2, color='black')),
            text=[f"{int(trade['trade_id'])}"], textfont=dict(size=10, color='black'),
            textposition='bottom center',
            name=f"Exit {int(trade['trade_id'])}", showlegend=False,
            hovertemplate=(
                f"Trade-ID: {int(trade['trade_id'])}<br>"
                f"Exit: {trade['Exit Time']}<br>"
                f"Exit-Preis: {trade['Exit Price']:.1f}<br>"
                f"Gewinn/Verlust: {trade['PnL_abs']:.1f} €<br>"
                f"Kapital nach Trade: {trade['Kapital nach Trade']:.1f} €<br>"
            )
        ))
        fig.add_trace(go.Scatter(
            x=[trade['Entry Time'], trade['Exit Time']],
            y=[trade['Entry Price'], trade['Exit Price']],
            mode='lines', line=dict(color=color, width=1.5, dash='dot'),
            showlegend=False
        ))
    fig.update_layout(title=strat, height=400)
    return fig

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(plot_single_candle(df, t1, strat1, COLORS_CANDLE[1]), use_container_width=True)
with c2:
    st.plotly_chart(plot_single_candle(df, t2, strat2, COLORS_CANDLE[2]), use_container_width=True)

# --- Zwei Equity-Kurven nebeneinander ---
eq1_times, eq1 = equity_curve(t1)
eq2_times, eq2 = equity_curve(t2)
e1, e2 = st.columns(2)
with e1:
    fig_eq1 = go.Figure(go.Scatter(x=eq1_times, y=eq1, mode='lines+markers', line=dict(color="green", width=3), marker=dict(size=5)))
    fig_eq1.update_layout(title=f"Equity: {strat1}", height=300)
    st.plotly_chart(fig_eq1, use_container_width=True)
with e2:
    fig_eq2 = go.Figure(go.Scatter(x=eq2_times, y=eq2, mode='lines+markers', line=dict(color="deepskyblue", width=3), marker=dict(size=5)))
    fig_eq2.update_layout(title=f"Equity: {strat2}", height=300)
    st.plotly_chart(fig_eq2, use_container_width=True)

# --- Kombi-Chart: beide Strategien im Kurs (schwarz/orange) und darunter beide Equities ---
st.markdown("### Direkter Vergleich: Trades & Equity-Verläufe")
fig_combo = make_subplots(
    rows=2, cols=1,
    subplot_titles=(f"Beide Strategien: Trades im Kurs-Chart", f"Beide Equity-Kurven"),
    vertical_spacing=0.12
)

fig_combo.add_trace(go.Candlestick(
    x=df['timestamp'],
    open=df['open'], high=df['high'], low=df['low'], close=df['close'],
    name="Kurs"
), row=1, col=1)

for i, (t, marker_color, strat) in enumerate([(t1, "black", strat1), (t2, "orange", strat2)], 1):
    for _, trade in t.iterrows():
        fig_combo.add_trace(go.Scatter(
            x=[trade['Entry Time']], y=[trade['Entry Price']],
            mode='markers+text',
            marker=dict(symbol='triangle-up', color=marker_color, size=13, line=dict(width=2, color='black')),
            text=[f"{int(trade['trade_id'])}"], textfont=dict(size=10, color=marker_color),
            textposition='top center',
            name=f"{strat} Entry {int(trade['trade_id'])}", showlegend=False,
            hovertemplate=(
                f"Trade-ID: {int(trade['trade_id'])}<br>"
                f"Entry: {trade['Entry Time']}<br>"
                f"Entry-Preis: {trade['Entry Price']:.1f}<br>"
                f"Einsatz: {trade['Einsatz']:.1f}<br>"
            )
        ), row=1, col=1)
        fig_combo.add_trace(go.Scatter(
            x=[trade['Exit Time']], y=[trade['Exit Price']],
            mode='markers+text',
            marker=dict(symbol='triangle-down', color=marker_color, size=13, line=dict(width=2, color='black')),
            text=[f"{int(trade['trade_id'])}"], textfont=dict(size=10, color=marker_color),
            textposition='bottom center',
            name=f"{strat} Exit {int(trade['trade_id'])}", showlegend=False,
            hovertemplate=(
                f"Trade-ID: {int(trade['trade_id'])}<br>"
                f"Exit: {trade['Exit Time']}<br>"
                f"Exit-Preis: {trade['Exit Price']:.1f}<br>"
                f"Gewinn/Verlust: {trade['PnL_abs']:.1f} €<br>"
                f"Kapital nach Trade: {trade['Kapital nach Trade']:.1f} €<br>"
            )
        ), row=1, col=1)
        fig_combo.add_trace(go.Scatter(
            x=[trade['Entry Time'], trade['Exit Time']],
            y=[trade['Entry Price'], trade['Exit Price']],
            mode='lines', line=dict(color=marker_color, width=1.5, dash='dot'),
            showlegend=False
        ), row=1, col=1)
# Beide Equity-Kurven mit Markern (schwarz/orange)
fig_combo.add_trace(go.Scatter(
    x=eq1_times, y=eq1, mode='lines+markers',
    line=dict(color="black", width=3), marker=dict(size=5, color="black"), name=f"{strat1} Equity"
), row=2, col=1)
fig_combo.add_trace(go.Scatter(
    x=eq2_times, y=eq2, mode='lines+markers',
    line=dict(color="orange", width=3), marker=dict(size=5, color="orange"), name=f"{strat2} Equity"
), row=2, col=1)
fig_combo.update_layout(height=900)

st.plotly_chart(fig_combo, use_container_width=True)
