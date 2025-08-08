import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from .metrics import rolling_sharpe_ratio, trade_cluster_summary

def create_advanced_dashboard(trades_df: pd.DataFrame, equity_df: pd.DataFrame, all_metrics: list, timestamp: str):
    """Bisheriges Übersichtsdashboard mit verschiedenen Performancecharts"""
    if trades_df.empty:
        return None
    equity_df = equity_df.copy()
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    portfolio_equity = equity_df.groupby('timestamp', as_index=False)['capital'].sum()
    portfolio_equity.rename(columns={'capital': 'portfolio_capital'}, inplace=True)

    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=(
            'Portfolio Equity (€)', 'Asset Performance', 'Monatliche Performance',
            'Trade Dauer', 'Profit Verteilung', 'Regime Performance',
            'Zeit Heatmap', 'Risk-Reward Scatter', 'Exit-Gründe',
            'Drawdown', 'Performance Attribution', 'Monte Carlo'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "histogram"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )

    # 1. Portfolio Equity
    fig.add_trace(
        go.Scatter(
            x=portfolio_equity['timestamp'],
            y=portfolio_equity['portfolio_capital'],
            mode='lines',
            name='Portfolio-Kapital (€)',
            line=dict(color='green', width=2),
            hovertemplate='%{x} Kapital: %{y:,.2f} €'
        ),
        row=1, col=1
    )

    # 2. Asset Performance
    asset_summary = trades_df.groupby('symbol').agg(
        total_profit=('profit', 'sum'),
        trade_count=('symbol', 'count')
    )
    fig.add_trace(
        go.Bar(
            x=asset_summary.index,
            y=asset_summary['total_profit'],
            name='Asset Profit (€)',
            marker_color='blue',
            text=asset_summary['trade_count'],
            hovertemplate='%{x} Gesamt: %{y:,.2f} € Trades: %{text}'
        ),
        row=1, col=2
    )

    # 3. Monatliche Performance
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    monthly = trades_df.groupby(trades_df['entry_time'].dt.to_period('M'))['profit'].sum()
    fig.add_trace(
        go.Bar(
            x=monthly.index.astype(str),
            y=monthly.values,
            name='Monatlich (€)',
            marker_color='orange',
            hovertemplate='%{x} Gewinn/Verlust: %{y:,.2f} €'
        ),
        row=1, col=3
    )

    # 4. Trade Dauer
    fig.add_trace(
        go.Histogram(
            x=trades_df['duration_hours'],
            name='Dauer (h)',
            nbinsx=30,
            marker_color='purple',
            hovertemplate='Dauer: %{x} h Anzahl: %{y}'
        ),
        row=2, col=1
    )

    # 5. Profit Verteilung
    colors = ['red' if x < 0 else 'green' for x in trades_df['profit']]
    fig.add_trace(
        go.Histogram(
            x=trades_df['profit'],
            name='Profit (€)',
            nbinsx=50,
            marker=dict(color=colors),
            hovertemplate='Profit: %{x:,.2f} € Anzahl: %{y}'
        ),
        row=2, col=2
    )

    # 6. Regime Performance
    regime_perf = trades_df.groupby('regime')['profit'].sum()
    fig.add_trace(
        go.Bar(
            x=regime_perf.index,
            y=regime_perf.values,
            name='Regime (€)',
            marker_color='teal',
            hovertemplate='%{x} Gewinn/Verlust: %{y:,.2f} €'
        ),
        row=2, col=3
    )

    # 7. Zeit Heatmap
    trades_df['weekday_name'] = trades_df['entry_time'].dt.day_name()
    trades_df['hour'] = trades_df['entry_time'].dt.hour
    heatmap_data = trades_df.pivot_table(
        values='profit',
        index=trades_df['entry_time'].dt.day_name(),
        columns=trades_df['entry_time'].dt.hour,
        aggfunc='sum'
    ).fillna(0)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex([d for d in day_order if d in heatmap_data.index])
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            name='Zeit Heatmap',
            hovertemplate='Stunde: %{x} Wochentag: %{y} Profit: %{z:,.2f} €'
        ),
        row=3, col=1
    )

    # 8. Risk-Reward Scatter
    fig.add_trace(
        go.Scatter(
            x=trades_df['risk_reward_ratio'],
            y=trades_df['profit'],
            mode='markers',
            name='Risk-Reward',
            marker=dict(color=np.where(trades_df['profit'] >= 0, 'green', 'red'), size=8),
            hovertemplate=(
                'Symbol: %{customdata[0]}<br>'
                'Entry: %{customdata[1]}<br>'
                'Exit: %{customdata[2]}<br>'
                'Einsatz: %{customdata[3]:,.2f} €<br>'
                'Profit: %{y:,.2f} €<br>'
                'R/R: %{x}<br>'
                'Exit-Grund: %{customdata[4]}<br>'
                'Regime: %{customdata[5]}'
            ),
            customdata=trades_df[['symbol', 'entry_time', 'exit_time', 'size_eur', 'exit_reason', 'regime']].values
        ),
        row=3, col=2
    )

    # 9. Exit-Gründe
    exit_reasons = trades_df.groupby('exit_reason')['profit'].sum()
    fig.add_trace(
        go.Bar(
            x=exit_reasons.index,
            y=exit_reasons.values,
            name='Exit-Gründe (€)',
            marker_color='indigo',
            hovertemplate='%{x} Gewinn/Verlust: %{y:,.2f} €'
        ),
        row=3, col=3
    )

    # 10. Drawdown
    running_max = portfolio_equity['portfolio_capital'].expanding().max()
    drawdown = (running_max - portfolio_equity['portfolio_capital']) / running_max * 100
    fig.add_trace(
        go.Scatter(
            x=portfolio_equity['timestamp'],
            y=drawdown,
            mode='lines',
            name='Drawdown (%)',
            line=dict(color='red'),
            hovertemplate='%{x} Drawdown: %{y:.2f}%'
        ),
        row=4, col=1
    )

    # 11. Performance Attribution (Pyramid Levels)
    if 'pyramid_levels' in trades_df.columns:
        pyramid_perf = trades_df.groupby('pyramid_levels')['profit'].sum()
        fig.add_trace(
            go.Bar(
                x=pyramid_perf.index,
                y=pyramid_perf.values,
                name='Pyramid Levels (€)',
                marker_color='brown',
                hovertemplate='Level: %{x} Profit: %{y:,.2f} €'
            ),
            row=4, col=2
        )

    # 12. Monte Carlo (wenn mehr als 10 Trades)
    if len(trades_df) > 10 and 'return_pct' in trades_df.columns:
        returns = trades_df['return_pct'].dropna()
        avg_return = returns.mean()
        std_return = returns.std()
        mc_values = [
            10000 * (1 + np.random.normal(avg_return, std_return, len(returns)) / 100).prod()
            for _ in range(1000)
        ]
        fig.add_trace(
            go.Histogram(
                x=mc_values,
                name='Monte Carlo (€)',
                nbinsx=20,
                marker_color='green',
                hovertemplate='Endkapital: %{x:,.2f} € Anzahl: %{y}'
            ),
            row=4, col=3
        )

    fig.update_layout(
        title=f"💰 Enhanced Portfolio Dashboard | {timestamp}",
        height=1600,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )
    html_file = f"results/enhanced_portfolio_dashboard_{timestamp}.html"
    fig.write_html(html_file)
    return html_file


def create_trade_marker_chart(trades_df: pd.DataFrame, equity_df: pd.DataFrame, timestamp: str):
    """Separater Chart Portfolio Equity mit Entry-/Exit-Marker (Dreiecke), Linien & Hoverinfos."""
    if trades_df.empty or equity_df.empty:
        return None
    equity_df = equity_df.copy()
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

    fig = go.Figure()

    # Portfolio-Kapitalverlauf Linie
    fig.add_trace(go.Scatter(
        x=equity_df["timestamp"],
        y=equity_df["capital"],
        mode="lines",
        name="Kapitalverlauf",
        line=dict(color="black", width=1.3)
    ))

    # Verbindungslinien und Marker
    for _, tr in trades_df.iterrows():
        color = "green" if tr["profit"] >= 0 else "red"

        # Linie Entry → Exit Kapital
        fig.add_trace(go.Scatter(
            x=[tr["entry_time"], tr["exit_time"]],
            y=[tr["capital_before"], tr["capital_after"]],
            mode="lines",
            line=dict(color=color, width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        ))

        # Entry Marker (Dreieck oben)
        fig.add_trace(go.Scatter(
            x=[tr["entry_time"]],
            y=[tr["capital_before"]],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=16, color=color, line=dict(width=1, color='black')),
            text=[str(tr["trade_nr"])],
            textposition="top center",
            showlegend=False,
            hovertemplate=(
                f"Trade #{tr['trade_nr']}<br>"
                f"{tr['symbol']}<br>"
                f"Entry: {tr['entry_time'].strftime('%d.%m.%Y %H:%M')}<br>"
                f"Kapitaleinsatz: {tr['size_eur']:.2f} €<br>"
                f"Signal: {tr.get('entry_reason','')}<br>"
            )
        ))

        # Exit Marker (Dreieck unten)
        fig.add_trace(go.Scatter(
            x=[tr["exit_time"]],
            y=[tr["capital_after"]],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=16, color=color, line=dict(width=1, color='black')),
            text=[str(tr["trade_nr"])],
            textposition="bottom center",
            showlegend=False,
            hovertemplate=(
                f"Trade #{tr['trade_nr']}<br>"
                f"{tr['symbol']}<br>"
                f"Exit: {tr['exit_time'].strftime('%d.%m.%Y %H:%M')}<br>"
                f"Kapitaleinsatz: {tr['size_eur']:.2f} €<br>"
                f"Exitgrund: {tr['exit_reason']}<br>"
                f"Gewinn/Verlust: {tr['profit']:.2f} €<br>"
            )
        ))

    fig.update_layout(
        title="Portfolio-Kapitalverlauf mit Entry/Exit-Markern",
        xaxis_title="Zeit",
        yaxis_title="Kapital (€)",
        template="plotly_white"
    )
    html_file = f"results/portfolio_trades_markers_{timestamp}.html"
    fig.write_html(html_file)
    return html_file


def create_detailed_dashboard(trades_df: pd.DataFrame, equity_df: pd.DataFrame, all_metrics: list, timestamp: str):
    """
    Erweiterte Visualisierung mit Rolling Sharpe, Cluster-Heatmap,
    monatlichem Profit, Profitverteilung – Ergänzung zum Basis-Dashboard.
    """
    if trades_df.empty or equity_df.empty:
        return None

    equity_df = equity_df.copy()
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    portfolio_equity = equity_df.groupby('timestamp', as_index=False)['capital'].sum()
    portfolio_equity.rename(columns={'capital': 'portfolio_capital'}, inplace=True)

    trades_df = trades_df.copy()
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

    rolling_sharpe = rolling_sharpe_ratio(equity_df, window=60)
    cluster_df = trade_cluster_summary(trades_df, bins=3)

    pivot_heatmap = cluster_df.pivot_table(
        values='profit_sum',
        index='regime',
        columns='volatility_class',
        fill_value=0
    )

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Portfolio-Kapitalverlauf (€)",
            "Rolling Sharpe Ratio (60 Perioden)",
            "Asset Performance (Gesamtprofit)",
            "Cluster Heatmap Profit nach Regime & Volatility",
            "Monatliche Performance",
            "Profit Verteilung"
        )
    )

    # Portfolio-Kapitalverlauf
    fig.add_trace(go.Scatter(
        x=portfolio_equity['timestamp'],
        y=portfolio_equity['portfolio_capital'],
        mode='lines',
        name='Portfolio-Kapital (€)',
        line=dict(color='green', width=2)
    ), row=1, col=1)

    # Rolling Sharpe Ratio
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        mode='lines',
        name='Rolling Sharpe Ratio',
        line=dict(color='orange', width=2)
    ), row=1, col=2)

    # Asset Performance
    asset_perf = trades_df.groupby('symbol').agg(total_profit=('profit', 'sum')).reset_index()
    fig.add_trace(go.Bar(
        x=asset_perf['symbol'],
        y=asset_perf['total_profit'],
        name='Asset Profit (€)',
        marker_color='blue'
    ), row=1, col=3)

    # Cluster Heatmap
    if not pivot_heatmap.empty:
        fig.add_trace(go.Heatmap(
            z=pivot_heatmap.values,
            x=pivot_heatmap.columns,
            y=pivot_heatmap.index,
            colorscale='RdYlGn',
            colorbar=dict(title="Gesamtprofit (€)"),
            name="Profit Cluster"
        ), row=2, col=1)

    # Monatliche Performance
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    monthly_profit = trades_df.groupby('month')['profit'].sum()
    fig.add_trace(go.Bar(
        x=monthly_profit.index.astype(str),
        y=monthly_profit.values,
        name="Monatliche Performance",
        marker_color='purple'
    ), row=2, col=2)

    # Profit Verteilung
    colors = ['green' if p > 0 else 'red' for p in trades_df['profit']]
    fig.add_trace(go.Histogram(
        x=trades_df['profit'],
        nbinsx=50,
        marker_color=colors,
        name="Profit Verteilung"
    ), row=2, col=3)

    fig.update_layout(
        height=800,
        width=1200,
        title_text=f"🔥 Detailliertes Trading-Dashboard | {timestamp}",
        template='plotly_white',
        hovermode='x unified'
    )

    html_file = f"results/detailed_trading_dashboard_{timestamp}.html"
    fig.write_html(html_file)
    return html_file
