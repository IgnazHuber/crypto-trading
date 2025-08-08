import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def compare_strategies_html(df, trades_dict, s1, s2, output_html):
    t1 = trades_dict[s1]
    t2 = trades_dict[s2]
    def equity_curve(trades):
        if trades.empty:
            return [0], [0]
        trades_sorted = trades.sort_values('Exit Time')
        times = [trades_sorted.iloc[0]['Entry Time']]
        capitals = [trades_sorted.iloc[0]['Kapital nach Trade'] - trades_sorted.iloc[0]['PnL_abs']]
        for i, t in trades_sorted.iterrows():
            times.append(t['Exit Time'])
            capitals.append(t['Kapital nach Trade'])
        return times, capitals

    n1, n2 = len(t1), len(t2)
    p1, p2 = t1['PnL_abs'].sum() if n1 else 0, t2['PnL_abs'].sum() if n2 else 0
    pct1 = p1 / t1['Einsatz'].sum() * 100 if n1 else 0
    pct2 = p2 / t2['Einsatz'].sum() * 100 if n2 else 0

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(f"{s1} Candles", f"{s2} Candles", f"{s1} Equity", f"{s2} Equity"),
        shared_xaxes='columns', vertical_spacing=0.08, horizontal_spacing=0.08
    )

    for i, (s, t) in enumerate([(s1, t1), (s2, t2)]):
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name="Kurs"
        ), row=1, col=i+1)
        for _, trade in t.iterrows():
            color = "green" if trade["PnL_abs"] >= 0 else "red"
            fig.add_trace(go.Scatter(
                x=[trade['Entry Time']], y=[trade['Entry Price']],
                mode='markers',
                marker=dict(symbol='triangle-up', color=color, size=14, line=dict(width=2, color='black')),
                name=f"Entry {int(trade['trade_id'])}", showlegend=False
            ), row=1, col=i+1)
            fig.add_trace(go.Scatter(
                x=[trade['Exit Time']], y=[trade['Exit Price']],
                mode='markers',
                marker=dict(symbol='triangle-down', color=color, size=14, line=dict(width=2, color='black')),
                name=f"Exit {int(trade['trade_id'])}", showlegend=False
            ), row=1, col=i+1)
            fig.add_trace(go.Scatter(
                x=[trade['Entry Time'], trade['Exit Time']],
                y=[trade['Entry Price'], trade['Exit Price']],
                mode='lines', line=dict(color=color, width=2, dash='dot'),
                showlegend=False
            ), row=1, col=i+1)
        # Equity
        times, capitals = equity_curve(t)
        fig.add_trace(go.Scatter(
            x=times, y=[round(x,1) for x in capitals],
            mode='lines+markers', line=dict(width=3), marker=dict(size=6),
            name=f"{s} Kapital", showlegend=False
        ), row=2, col=i+1)
    # Info oben drüber
    fig.update_layout(
        title={
            "text": (
                f"Vergleich: {s1} (Trades: {n1}, G/V: {p1:.1f} €/{pct1:.1f} %) "
                f"vs. {s2} (Trades: {n2}, G/V: {p2:.1f} €/{pct2:.1f} %)"
            ),
            "x": 0.5, "y": 0.96, "xanchor": "center", "yanchor": "top"
        },
        height=900,
        template="plotly_white"
    )
    fig.write_html(output_html)
    print(f"[plotly] Vergleichschart gespeichert: {output_html}")

# --- Beispielaufruf ---
if __name__ == "__main__":
    # Hier df (aus Parquet), trades_dict laden (z.B. aus pickle oder erneut berechnen)
    pass  # Siehe Dokumentation, wie du es laden willst
