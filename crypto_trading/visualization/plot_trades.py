# visualization/plot_trades.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_strategy_chart_with_equity(df, trades, strategy_name, filename):
    n_trades = len(trades)
    total_pnl = trades['PnL_abs'].sum() if n_trades > 0 else 0
    total_einsatz = trades['Einsatz'].sum() if n_trades > 0 else 1
    total_pct = total_pnl / total_einsatz * 100 if n_trades > 0 else 0

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
        subplot_titles=(
            f"{strategy_name}",
            "Portfolioverlauf (Equity Curve)"
        )
    )

    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name="Kurs"
    ), row=1, col=1)

    for _, t in trades.iterrows():
        color = "green" if t["PnL_abs"] >= 0 else "red"
        fig.add_trace(go.Scatter(
            x=[t['Entry Time']], y=[t['Entry Price']],
            mode='markers+text',
            marker=dict(symbol='triangle-up', color=color, size=18, line=dict(width=2, color='black')),
            text=[f"{int(t['trade_id'])}"],
            textfont=dict(size=11),
            textposition='top center',
            name=f"Entry {int(t['trade_id'])}",
            hovertemplate=(
                f"Trade-ID: {int(t['trade_id'])}<br>"
                f"Krypto: {t['asset']}<br>"
                f"Strategie: {t['Strategy']}<br>"
                f"Entry: {t['Entry Time']}<br>"
                f"Entry-Preis: {t['Entry Price']:.1f}<br>"
                f"Einsatz: {t['Einsatz']:.1f}<br>"
            ),
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[t['Exit Time']], y=[t['Exit Price']],
            mode='markers+text',
            marker=dict(symbol='triangle-down', color=color, size=18, line=dict(width=2, color='black')),
            text=[f"{int(t['trade_id'])}"],
            textfont=dict(size=11),
            textposition='bottom center',
            name=f"Exit {int(t['trade_id'])}",
            hovertemplate=(
                f"Trade-ID: {int(t['trade_id'])}<br>"
                f"Krypto: {t['asset']}<br>"
                f"Strategie: {t['Strategy']}<br>"
                f"Exit: {t['Exit Time']}<br>"
                f"Exit-Preis: {t['Exit Price']:.1f}<br>"
                f"Gewinn/Verlust: {t['PnL_abs']:.1f} €<br>"
                f"Gewinn/Verlust: {t['PnL_pct']:.1f}%<br>"
                f"Kapital nach Trade: {t['Kapital nach Trade']:.1f} €<br>"
            ),
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[t['Entry Time'], t['Exit Time']],
            y=[t['Entry Price'], t['Exit Price']],
            mode='lines',
            line=dict(color=color, width=2, dash='dot'),
            hoverinfo='skip',
            showlegend=False
        ), row=1, col=1)

    # Equity-Kurve unten
    if not trades.empty:
        trades_sorted = trades.sort_values('Exit Time')
        times = [trades_sorted.iloc[0]['Entry Time']]
        capitals = [trades_sorted.iloc[0]['Kapital nach Trade'] - trades_sorted.iloc[0]['PnL_abs']]
        for i, t in trades_sorted.iterrows():
            times.append(t['Exit Time'])
            capitals.append(t['Kapital nach Trade'])
        capitals = [round(x,1) for x in capitals]
        fig.add_trace(go.Scatter(
            x=times, y=capitals,
            mode='lines+markers',
            line=dict(color="blue", width=3),
            marker=dict(size=6),
            name="Kapital",
            hovertemplate="Zeit: %{x}<br>Kapital: %{y:.1f} €",
            showlegend=False
        ), row=2, col=1)

    fig.update_yaxes(title_text="Preis", row=1, col=1, rangemode="normal")
    fig.update_yaxes(title_text="Kapital (€)", row=2, col=1, rangemode="normal")
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(
        title={
            "text": (
                f"{strategy_name}<br>"
                f"<span style='font-size:15px;color:#888;'>Trades: {n_trades}, "
                f"Gesamtgewinn/-verlust: {total_pnl:.1f} € ({total_pct:.1f} %)</span>"
            ),
            "x": 0.5, "y": 0.96,
            "xanchor": "center", "yanchor": "top"
        },
        hovermode='closest',
        height=850,
        template="plotly_white"
    )
    fig.write_html(filename)
    print(f"[plotly] Chart mit Portfolioverlauf gespeichert: {filename}")
