# analysis/mod_plots.py

import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def print_and_show_performance(trades, fig=None):
    if trades.empty:
        print("Keine Trades im Backtest!")
        return
    n_trades = len(trades)
    total_pnl = trades["PnL_abs"].sum()
    avg_pnl = trades["PnL_abs"].mean()
    winrate = 100 * (trades["PnL_abs"] > 0).sum() / n_trades if n_trades > 0 else 0
    start_capital = trades["Kapital nach Trade"].iloc[0] - trades["PnL_abs"].iloc[0] if n_trades > 0 else 0
    end_capital = trades["Kapital nach Trade"].iloc[-1] if n_trades > 0 else 0
    max_drawdown = None
    if "Kapital nach Trade" in trades.columns:
        roll_max = trades["Kapital nach Trade"].cummax()
        drawdown = trades["Kapital nach Trade"] / roll_max - 1.0
        max_drawdown = drawdown.min()

    print("\n===== Backtest Performance =====")
    print(f"Anzahl Trades        : {n_trades}")
    print(f"Gesamt-PnL (Summe)  : {total_pnl:.2f}")
    print(f"Durchschnittlicher PnL: {avg_pnl:.2f}")
    print(f"Winrate (%)         : {winrate:.2f}")
    print(f"Startkapital        : {start_capital:.2f}")
    print(f"Endkapital          : {end_capital:.2f}")
    if max_drawdown is not None:
        print(f"Max Drawdown        : {max_drawdown:.2%}")
    print("================================\n")

    if fig is not None:
        max_drawdown_str = f"{max_drawdown:.2%}" if max_drawdown is not None else "-"
        kpi_text = (
            f"Anzahl Trades: {n_trades}<br>"
            f"Gesamt-PnL: {total_pnl:.2f}<br>"
            f"Durchschn. PnL: {avg_pnl:.2f}<br>"
            f"Winrate: {winrate:.2f}%<br>"
            f"Startkapital: {start_capital:.2f}<br>"
            f"Endkapital: {end_capital:.2f}<br>"
            f"Max Drawdown: {max_drawdown_str}"
        )
        fig.add_annotation(
            text=kpi_text,
            xref="paper", yref="paper", x=1.0, y=-0.28, showarrow=False,
            align="left", font=dict(size=14, color="black"),
            bordercolor="gray", borderwidth=1, borderpad=6,
            bgcolor="white", opacity=0.85
        )


def plot_dual_chart_with_markers(df, trades, html_path, strat_name="Meta_Score", asset=None):
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    if trades.empty:
        print("[plot_dual_chart_with_markers] Keine Trades, kein Chart erzeugt.")
        return

    if asset is None:
        asset = trades['Asset'].iloc[0] if 'Asset' in trades.columns else "Asset"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Kursverlauf mit Trades", "Equity-Kurve mit Trades"),
        vertical_spacing=0.08, row_heights=[0.6, 0.4]
    )

    # Fallback X-Achse
    x_axis = df['timestamp'] if 'timestamp' in df.columns else df.index

    # Preis-Kurve
    fig.add_trace(go.Scatter(
        x=x_axis, y=df['close'], mode="lines", name="Kurs", line=dict(color="gray")
    ), row=1, col=1)

    # --- Hovertexte für Entry & Exit ---
    entry_hover = [
        (f"<b>Trade-ID:</b> {row['Trade-ID']}<br>"
         f"<b>Entry:</b> {row['Entry Time']}<br>"
         f"<b>Entry-Preis:</b> {row['Entry Price']}<br>"
         f"<b>Einsatz:</b> {row['Einsatz']}<br>"
         f"<b>Regime:</b> {row['market_regime']}<br>"
         f"<b>Score:</b> {row['score_entry']}<br>"
         f"<b>Trigger:</b> {row['entry_trigger']}<br>")
        for _, row in trades.iterrows()
    ]
    exit_hover = [
        (f"<b>Trade-ID:</b> {row['Trade-ID']}<br>"
         f"<b>Exit:</b> {row['Exit Time']}<br>"
         f"<b>Exit-Preis:</b> {row['Exit Price']}<br>"
         f"<b>Trade-Dauer:</b> {row['trade_duration']}<br>"
         f"<b>PnL:</b> {row['PnL_abs']} ({row['PnL_pct']}%)<br>"
         f"<b>Equity:</b> {row['Kapital nach Trade']}<br>"
         f"<b>Schließungsgrund:</b> {row['close_reason']}<br>"
         f"<b>SL Hit:</b> {row['sl_hit']} / <b>TP Hit:</b> {row['tp_hit']}<br>")
        for _, row in trades.iterrows()
    ]

    # --- Entry-Marker ---
    fig.add_trace(go.Scatter(
        x=trades["Entry Time"], y=trades["Entry Price"],
        mode="markers+text",
        marker=dict(color="green", symbol="triangle-up", size=12),
        name="Entry",
        text=trades["Trade-ID"].astype(str),
        textposition="top center",
        textfont=dict(size=11, color="black"),
        hoverinfo="text", hovertext=entry_hover
    ), row=1, col=1)

    # --- Exit-Marker ---
    fig.add_trace(go.Scatter(
        x=trades["Exit Time"], y=trades["Exit Price"],
        mode="markers+text",
        marker=dict(color="red", symbol="triangle-down", size=12),
        name="Exit",
        text=trades["Trade-ID"].astype(str),
        textposition="top center",
        textfont=dict(size=11, color="red"),
        hoverinfo="text", hovertext=exit_hover
    ), row=1, col=1)

    # --- Verbindungslinien Entry → Exit ---
    for _, row in trades.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["Entry Time"], row["Exit Time"]],
            y=[row["Entry Price"], row["Exit Price"]],
            mode="lines",
            line=dict(color="black", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=False
        ), row=1, col=1)

    # --- Equity-Kurve mit Marker + Hover ---
    equity_hover = [
        (f"<b>Trade-ID:</b> {row['Trade-ID']}<br>"
         f"<b>Exit:</b> {row['Exit Time']}<br>"
         f"<b>Equity nach Trade:</b> {row['Kapital nach Trade']}<br>"
         f"<b>PnL:</b> {row['PnL_abs']} ({row['PnL_pct']}%)<br>")
        for _, row in trades.iterrows()
    ]
    fig.add_trace(go.Scatter(
        x=trades["Exit Time"], y=trades["Kapital nach Trade"],
        mode="lines+markers+text",
        marker=dict(color="blue", size=8, symbol="circle"),
        name="Equity-Kurve",
        text=trades["Trade-ID"].astype(str),
        textposition="top center",
        hoverinfo="text", hovertext=equity_hover
    ), row=2, col=1)

    # Layout
    fig.update_layout(
        title=f"{asset} Backtest: {strat_name}",
        xaxis_title="Zeit", yaxis_title="Preis",
        xaxis2_title="Zeit", yaxis2_title="Kapital",
        hovermode="closest",
        legend=dict(orientation="h"),
        height=900
    )

    # Performance-Box
    print_and_show_performance(trades, fig)

    # HTML schreiben
    fig.write_html(html_path)
    print(f"[plot_dual_chart_with_markers] Plotly-HTML exportiert: {html_path}")
import plotly.graph_objects as go

def export_png_equity(trades_df, path):
    """Exportiert Equity-Kurve als PNG."""
    if trades_df.empty or "capital" not in trades_df:
        print("[Plot] Keine Trades für Equity-Kurve.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=trades_df["capital"],
        mode="lines",
        name="Equity"
    ))
    fig.update_layout(title="Equity-Kurve", xaxis_title="Trade #", yaxis_title="Kapital")
    fig.write_image(path)
    print(f"[Plot] Equity-Kurve gespeichert: {path}")
