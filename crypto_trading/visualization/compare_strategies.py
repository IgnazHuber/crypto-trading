# d:\Projekte\crypto_trading\crypto_trading\visualization\compare_strategies_dash.py

import os
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from crypto_trading.config.settings import RESULTS_DIR, safe_sheet_name
from crypto_trading.core.indicators import add_indicators  # falls needed für spätere Varianten

# ---- Lade Daten für die App (Excel mit allen Trades, Namen extrahieren) ----
EXCEL_PATH = os.path.join(RESULTS_DIR, "btc_candlestick_backtest_regime.xlsx")
df_all = pd.read_excel(EXCEL_PATH, sheet_name=None)
trades_dict = {k: v for k, v in df_all.items() if k not in ("KPIs", "Alle Trades")}
# Timestamp zu pd.Timestamp (wichtig für Plotly)
for trades in trades_dict.values():
    for c in ["Entry Time", "Exit Time"]:
        if trades[c].dtype == "object":
            trades[c] = pd.to_datetime(trades[c])

# Lade Candlestick-Daten (Dein Parquet)
from crypto_trading.config.settings import PARQUET_PATH
df = pd.read_parquet(PARQUET_PATH)
if 'timestamp' in df and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# ---- Farben wie gehabt ----
COLORS = {
    1: {"win": "green", "loss": "red", "equity": "green"},
    2: {"win": "deepskyblue", "loss": "blue", "equity": "deepskyblue"}
}

# ---- Dash App ----
app = Dash(__name__)
app.title = "Strategie-Vergleich"

sheetnames = list(trades_dict.keys())

app.layout = html.Div([
    html.H2("Strategie-Vergleich (interaktiv)"),
    html.Div([
        html.Label("Strategie 1:"), 
        dcc.Dropdown(
            id='strat1-dropdown',
            options=[{'label': s, 'value': s} for s in sheetnames],
            value=sheetnames[0]
        ),
        html.Label("Strategie 2:"),
        dcc.Dropdown(
            id='strat2-dropdown',
            options=[{'label': s, 'value': s} for s in sheetnames],
            value=sheetnames[1] if len(sheetnames)>1 else sheetnames[0]
        )
    ], style={"width": "60%", "margin": "auto", "display": "flex", "gap": "30px"}),
    html.Br(),
    dcc.Loading(dcc.Graph(id="compare-graph"), type="circle")
], style={"maxWidth": "1600px", "margin": "auto"})

@app.callback(
    Output("compare-graph", "figure"),
    Input("strat1-dropdown", "value"),
    Input("strat2-dropdown", "value")
)
def update_figure(strat1, strat2):
    t1 = trades_dict[strat1]
    t2 = trades_dict[strat2]
    n1, n2 = len(t1), len(t2)
    p1, p2 = t1['PnL_abs'].sum() if n1 else 0, t2['PnL_abs'].sum() if n2 else 0
    pct1 = p1 / t1['Einsatz'].sum() * 100 if n1 else 0
    pct2 = p2 / t2['Einsatz'].sum() * 100 if n2 else 0

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

if __name__ == "__main__":
    app.run(debug=True, port=8055)
