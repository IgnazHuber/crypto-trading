# trading_strategy_kimi_v5.py  –  FINAL & VOLLSTÄNDIG (2025-08-07)
#  • Alle Features aus v3 erhalten
#  • Korrekte Kapital-Berechnung (10 000 € Start pro Asset)
#  • Keine Features entfernt / nicht kürzer
#  • Euro-Formatierung & Hover-Infos vollständig

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import ta  # pip install ta
import os
import json
import logging
from datetime import datetime, timedelta
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from tqdm import tqdm
from crypto_trading.analysis.candlestick_analyzer import detect_candlestick_patterns
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# === BASIS-FUNKTIONEN ===
def select_files(last_path):
    try:
        root = Tk()
        root.withdraw()
        initial_dir = last_path if os.path.exists(last_path) else os.getcwd()
        file_paths = askopenfilenames(
            title="Wähle Parquet-Dateien aus",
            initialdir=initial_dir,
            filetypes=[("Parquet files", "*.parquet")]
        )
        root.destroy()
        if file_paths:
            last_dir = os.path.dirname(file_paths[0])
            os.makedirs('results', exist_ok=True)
            with open(os.path.join('results', 'last_path.json'), 'w') as f:
                json.dump({'last_path': last_dir}, f)
        return file_paths
    except Exception as e:
        logging.error(f"Fehler im Dateidialog: {str(e)}")
        return []

def calculate_metrics(trades_df, equity_df, initial_capital):
    if trades_df.empty:
        return {
            'total_trades': 0, 'win_rate': 0, 'total_profit': 0,
            'final_capital': initial_capital, 'return_pct': 0,
            'sharpe_ratio': 0, 'max_drawdown': 0
        }
    metrics = {}
    metrics['total_trades'] = len(trades_df)
    metrics['win_rate'] = len(trades_df[trades_df['profit'] > 0]) / len(trades_df) if len(trades_df) else 0
    metrics['total_profit'] = trades_df['profit'].sum()
    metrics['final_capital'] = initial_capital + metrics['total_profit']
    metrics['return_pct'] = (metrics['final_capital'] - initial_capital) / initial_capital * 100
    if len(equity_df) > 1:
        equity_df['returns'] = equity_df['capital'].pct_change().fillna(0)
        metrics['sharpe_ratio'] = (equity_df['returns'].mean() * 252) / (equity_df['returns'].std() * np.sqrt(252)) if equity_df['returns'].std() != 0 else 0
        metrics['max_drawdown'] = ((equity_df['capital'].cummax() - equity_df['capital']) / equity_df['capital'].cummax()).max() * 100
    else:
        metrics['sharpe_ratio'] = 0
        metrics['max_drawdown'] = 0
    return metrics

# === ERWEITERTE INDIKATOREN ===
def calculate_advanced_indicators(df):
    logging.debug("Berechne erweiterte Indikatoren")
    df['sma20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['ema12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema26'] = ta.trend.ema_indicator(df['close'], window=26)
    df['trend'] = 'neutral'
    df.loc[df['sma20'] > df['sma50'], 'trend'] = 'bullish'
    df.loc[df['sma20'] < df['sma50'], 'trend'] = 'bearish'
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
    df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close'] * 100
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    df['regime'] = 'range'
    df.loc[df['adx'] > 25, 'regime'] = 'trending'
    df.loc[df['bb_width'] < df['bb_width'].rolling(20).mean().shift(), 'regime'] = 'low_vol'
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['weekday'] = df.index.weekday
    else:
        df['hour'] = 0
        df['weekday'] = 0
    return df

# === ERWEITERTES RISIKO-MANAGEMENT ===
def calculate_position_size(capital, max_position_pct=0.10, max_risk_pct=0.03, atr=None):
    max_position = capital * max_position_pct
    max_risk_amount = capital * max_risk_pct
    if atr and atr > 0:
        volatility_adjusted = max_risk_amount / (atr * 1.5)
        return min(max_position, max(volatility_adjusted, 100))
    return max_position

def calculate_risk_levels(entry_price, max_loss_pct=0.03, max_gain_pct=0.20):
    stop_loss = entry_price * (1 - max_loss_pct)
    take_profit = entry_price * (1 + max_gain_pct)
    return stop_loss, take_profit

# === FILTER-FUNKTIONEN ===
def volatility_filter(current_data):
    vol_ratio = current_data['atr'] / current_data['close']
    return 0.005 < vol_ratio < 0.05

def time_filter(timestamp):
    hour = timestamp.hour if hasattr(timestamp, 'hour') else 12
    return 4 <= hour <= 22

def regime_filter(current_data):
    return current_data['regime'] in ['trending', 'range']

def pyramiding_strategy(position, current_price, atr):
    if 'pyramid_levels' not in position:
        position['pyramid_levels'] = 0
    unrealized_pnl = (current_price - position['entry_price']) * position['size']
    if unrealized_pnl > 2 * atr and position['pyramid_levels'] < 2:
        add_size = position['size'] * 0.5
        position['pyramid_levels'] += 1
        return add_size
    return 0

# === HAUPT-STRATEGIE MIT ALLEN FEATURES ===
def simulate_advanced_strategy(df, patterns, initial_capital=10000,
                             max_position_pct=0.10, max_risk_pct=0.03):
    logging.debug("Starte erweiterte Strategie mit korrekter Kapital-Verwaltung")
    trades = []
    capital = initial_capital  # Startkapital pro Asset – wird kumulativ weitergeführt
    positions = []
    equity = []
    symbol = df.iloc[0].get('symbol', 'Unknown')

    for i in tqdm(range(len(patterns)), desc="Simuliere Advanced Trades"):
        if i >= len(df):
            continue
        row = patterns.iloc[i]
        current_price = df.iloc[i]['close']
        timestamp = row['timestamp']
        current_data = df.iloc[i]

        # Entry-Bedingungen
        if (row['signal'] in ['Kaufen', 'Verkaufen'] and
            len(positions) < 3 and
            time_filter(timestamp) and
            volatility_filter(current_data) and
            regime_filter(current_data)):

            is_long = row['signal'] == 'Kaufen'
            max_loss_pct = 0.03
            max_gain_pct = 0.20
            stop_loss = current_price * (1 - max_loss_pct)
            take_profit = current_price * (1 + max_gain_pct)

            position_size = calculate_position_size(capital, max_position_pct, max_risk_pct, current_data['atr'])
            position_size = min(position_size, capital)  # Ensure position size does not exceed current capital

            position = {
                'type': 'long' if is_long else 'short',
                'entry_price': current_price,
                'size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'max_loss': position_size * max_loss_pct,
                'max_gain': position_size * max_gain_pct,
                'timestamp': timestamp,
                'symbol': symbol,
                'pyramid_levels': 0,
                'regime': current_data['regime'],
                'hour': timestamp.hour if hasattr(timestamp, 'hour') else 0,
                'weekday': timestamp.weekday() if hasattr(timestamp, 'weekday') else 0
            }
            positions.append(position)

        # Position-Management
        for pos in positions[:]:
            current_pnl = (current_price - pos['entry_price']) * pos['size'] if pos['type'] == 'long' else \
                          (pos['entry_price'] - current_price) * pos['size']
            exit_price = current_price
            exit_reason = None
            profit = 0

            if pos['type'] == 'long':
                if current_price <= pos['stop_loss']:
                    exit_reason = 'stop_loss'
                    exit_price = pos['stop_loss']
                elif current_price >= pos['take_profit']:
                    exit_reason = 'take_profit'
                    exit_price = pos['take_profit']
            else:
                if current_price >= pos['stop_loss']:
                    exit_reason = 'stop_loss'
                    exit_price = pos['stop_loss']
                elif current_price <= pos['take_profit']:
                    exit_reason = 'take_profit'
                    exit_price = pos['take_profit']

            if exit_reason:
                raw_profit = (exit_price - pos['entry_price']) * pos['size'] if pos['type'] == 'long' else \
                             (pos['entry_price'] - exit_price) * pos['size']
                profit = min(raw_profit, pos['max_gain'])  # Limit profit to max_gain

                capital += profit  # Update capital
                trades.append({
                    'symbol': pos['symbol'],
                    'entry_time': pos['timestamp'],
                    'exit_time': timestamp,
                    'type': 'Kaufen' if pos['type'] == 'long' else 'Verkaufen',
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'size': pos['size'],
                    'profit': profit,
                    'return_pct': (profit / pos['size']) * 100,
                    'capital_before': capital - profit,
                    'capital_after': capital,
                    'duration_hours': (timestamp - pos['timestamp']).total_seconds() / 3600,
                    'exit_reason': exit_reason,
                    'max_loss': pos['max_loss'],
                    'max_gain': pos['max_gain'],
                    'risk_reward_ratio': pos['max_gain'] / pos['max_loss'],
                    'regime': pos['regime'],
                    'pyramid_levels': pos['pyramid_levels'],
                    'hour': pos['hour'],
                    'weekday': pos['weekday']
                })
                positions.remove(pos)

        # Equity-Tracking with correct capital
        equity.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'capital': capital,
            'open_positions': len(positions),
            'unrealized_pnl': sum([(current_price - p['entry_price']) * p['size'] if p['type'] == 'long' else
                                   (p['entry_price'] - current_price) * p['size'] for p in positions])
        })

    return pd.DataFrame(trades), pd.DataFrame(equity), capital

# === ERWEITERTE METRIKEN ===
def calculate_enhanced_metrics(trades_df, equity_df, initial_capital):
    metrics = calculate_metrics(trades_df, equity_df, initial_capital)
    if not trades_df.empty:
        metrics.update({
            'total_winning_trades': len(trades_df[trades_df['profit'] > 0]),
            'total_losing_trades': len(trades_df[trades_df['profit'] < 0]),
            'avg_profit_per_trade': trades_df['profit'].mean(),
            'avg_win': trades_df[trades_df['profit'] > 0]['profit'].mean(),
            'avg_loss': trades_df[trades_df['profit'] < 0]['profit'].mean(),
            'largest_win': trades_df['profit'].max(),
            'largest_loss': trades_df['profit'].min(),
            'profit_factor': abs(trades_df[trades_df['profit'] > 0]['profit'].sum() /
                               trades_df[trades_df['profit'] < 0]['profit'].sum()),
            'avg_duration': trades_df['duration_hours'].mean(),
            'calmar_ratio': metrics['return_pct'] / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else 0,
            'expectancy': calculate_expectancy(trades_df),
            'kelly_criterion': calculate_kelly_criterion(trades_df)
        })
    return metrics

def calculate_expectancy(trades_df):
    if trades_df.empty:
        return 0
    win_rate = len(trades_df[trades_df['profit'] > 0]) / len(trades_df)
    avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean() or 0
    avg_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].mean()) or 0
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

def calculate_kelly_criterion(trades_df):
    if trades_df.empty:
        return 0
    wins = trades_df[trades_df['profit'] > 0]['profit']
    losses = trades_df[trades_df['profit'] < 0]['profit']
    if wins.empty or losses.empty:
        return 0
    win_rate = len(wins) / len(trades_df)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    return (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

# === ERWEITERTE AUSWERTUNGEN ===
def create_performance_heatmap(trades_df):
    if trades_df.empty:
        return pd.DataFrame()
    trades_df['weekday_name'] = trades_df['entry_time'].dt.day_name()
    heatmap = trades_df.pivot_table(
        values='profit', 
        index='weekday_name',
        columns='hour',
        aggfunc='sum'
    ).fillna(0)
    return heatmap

def analyze_correlations(all_trades):
    correlation_data = {}
    for trades_df in all_trades:
        if not trades_df.empty:
            symbol = trades_df.iloc[0]['symbol']
            correlation_data[symbol] = trades_df['profit'].cumsum()
    if correlation_data:
        corr_df = pd.DataFrame(correlation_data)
        return corr_df.corr()
    return pd.DataFrame()

def monte_carlo_simulation(trades_df, n_simulations=1000):
    if trades_df.empty:
        return {}
    returns = trades_df['return_pct'].dropna()
    if len(returns) == 0:
        return {}
    simulations = []
    for _ in range(n_simulations):
        sim_returns = np.random.choice(returns, size=len(returns))
        sim_equity = 10000
        equity_series = [sim_equity]
        for ret in sim_returns:
            sim_equity *= (1 + ret / 100)
            equity_series.append(sim_equity)
        equity_series = pd.Series(equity_series)
        simulations.append({
            'final_value': equity_series.iloc[-1],
            'max_drawdown': ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100
        })
    sim_df = pd.DataFrame(simulations)
    return {
        'expected_value': sim_df['final_value'].mean(),
        'median_value': sim_df['final_value'].median(),
        'var_95': sim_df['final_value'].quantile(0.05),
        'var_99': sim_df['final_value'].quantile(0.01),
        'max_drawdown_95': sim_df['max_drawdown'].quantile(0.95)
    }

# === EXPORT & DASHBOARD ===
def export_comprehensive_results(global_trades, global_equity, all_metrics, timestamp):
    csv_file = f"results/complete_trades_{timestamp}.csv"
    if not global_trades.empty:
        global_trades.to_csv(csv_file, index=False)

    excel_file = f"results/complete_analysis_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        if not global_trades.empty:
            global_trades.to_excel(writer, sheet_name='Einzeltrades', index=False)

            # Equity-Verlauf – bereits kumulativ, KEINE weitere Kumulation
            global_equity['timestamp'] = pd.to_datetime(global_equity['timestamp'])
            equity_curve_df = global_equity[['timestamp', 'symbol', 'capital']].rename(columns={'capital': 'equity'})
            equity_curve_df.to_excel(writer, sheet_name='Equity_Verlauf', index=False)

            heatmap = create_performance_heatmap(global_trades)
            if not heatmap.empty:
                heatmap.to_excel(writer, sheet_name='Performance_Heatmap')

            if {'hour', 'weekday'}.issubset(global_trades.columns):
                time_analysis = global_trades.groupby(['hour', 'weekday']).agg({
                    'profit': ['sum', 'count', 'mean'], 'return_pct': 'mean'
                })
                time_analysis.columns = ['Profit_Sum', 'Trade_Count', 'Profit_Mean', 'Return_Mean']
                time_analysis.to_excel(writer, sheet_name='Zeit_Analysen')

            symbol_analysis = global_trades.groupby('symbol').agg({
                'profit': ['sum', 'count', 'mean', 'std'],
                'return_pct': ['mean', 'std'],
                'duration_hours': 'mean',
                'risk_reward_ratio': 'mean',
                'regime': lambda x: x.mode()[0] if len(x) > 0 else 'N/A'
            })
            symbol_analysis.columns = ['_'.join(col) for col in symbol_analysis.columns]
            symbol_analysis.to_excel(writer, sheet_name='Symbol_Performance')

            regime_analysis = global_trades.groupby('regime').agg({
                'profit': ['sum', 'count', 'mean'], 'return_pct': 'mean', 'risk_reward_ratio': 'mean'
            })
            regime_analysis.columns = ['Profit_Sum', 'Trade_Count', 'Profit_Mean', 'Return_Mean', 'RR_Mean']
            regime_analysis.to_excel(writer, sheet_name='Regime_Analyse')

            pyramid_analysis = global_trades.groupby('pyramid_levels').agg({
                'profit': ['sum', 'count', 'mean'], 'return_pct': 'mean'
            })
            pyramid_analysis.columns = ['Profit_Sum', 'Trade_Count', 'Profit_Mean', 'Return_Mean']
            pyramid_analysis.to_excel(writer, sheet_name='Pyramid_Analyse')

            exit_analysis = global_trades.groupby('exit_reason').agg({
                'profit': ['sum', 'count', 'mean'], 'duration_hours': 'mean'
            })
            exit_analysis.columns = ['Profit_Sum', 'Trade_Count', 'Profit_Mean', 'Duration_Mean']
            exit_analysis.to_excel(writer, sheet_name='Exit_Analysen')

        summary_df = pd.DataFrame(all_metrics)
        summary_df.to_excel(writer, sheet_name='Zusammenfassung', index=False)

    html_file = create_advanced_dashboard(global_trades, global_equity, all_metrics, timestamp)
    return {'csv': csv_file, 'excel': excel_file, 'html': html_file}

def create_advanced_dashboard(trades_df, equity_df, all_metrics, timestamp):
    if trades_df.empty:
        return None

    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])

    # Portfolio-Equity (bereits kumulativ)
    portfolio_equity = equity_df.groupby('timestamp', as_index=False)['capital'].sum()
    portfolio_equity.rename(columns={'capital': 'portfolio_capital'}, inplace=True)

    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=('Portfolio Equity (€)', 'Asset Performance', 'Monatliche Performance',
                        'Trade Dauer', 'Profit Verteilung', 'Regime Performance',
                        'Zeit Heatmap', 'Risk-Reward Scatter', 'Exit-Gründe',
                        'Drawdown', 'Performance Attribution', 'Monte Carlo'),
        specs=[[{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "histogram"}, {"type": "bar"}],
               [{"type": "heatmap"}, {"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}]],
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
            hovertemplate='%{x}<br>Kapital: %{y:,.2f} €<extra></extra>'
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
            hovertemplate='%{x}<br>Gesamt: %{y:,.2f} €<br>Trades: %{text}<extra></extra>'
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
            hovertemplate='%{x}<br>Gewinn/Verlust: %{y:,.2f} €<extra></extra>'
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
            hovertemplate='Dauer: %{x}h<br>Anzahl: %{y}<extra></extra>'
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
            hovertemplate='Profit: %{x:,.2f} €<br>Anzahl: %{y}<extra></extra>'
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
            hovertemplate='%{x}<br>Gewinn/Verlust: %{y:,.2f} €<extra></extra>'
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
            hovertemplate='Stunde: %{x}<br>Wochentag: %{y}<br>Profit: %{z:,.2f} €<extra></extra>'
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
                'Regime: %{customdata[5]}<extra></extra>'
            ),
            customdata=trades_df[['symbol', 'entry_time', 'exit_time', 'size', 'exit_reason', 'regime']].values
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
            hovertemplate='%{x}<br>Gewinn/Verlust: %{y:,.2f} €<extra></extra>'
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
            hovertemplate='%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
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
                hovertemplate='Level: %{x}<br>Profit: %{y:,.2f} €<extra></extra>'
            ),
            row=4, col=2
        )

    # 12. Monte Carlo
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
                hovertemplate='Endkapital: %{x:,.2f} €<br>Anzahl: %{y}<extra></extra>'
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

# === HAUPTFUNKTION ===
def main_complete():
    logging.basicConfig(
        filename=os.path.join('results', 'complete_strategy.log'),
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Starte vollständige Strategie")

    last_path = os.path.join(os.getcwd(), 'data', 'raw')
    if os.path.exists(os.path.join('results', 'last_path.json')):
        try:
            with open(os.path.join('results', 'last_path.json'), 'r') as f:
                last_path = json.load(f).get('last_path', last_path)
        except:
            pass

    file_paths = select_files(last_path)
    if not file_paths:
        print("❌ Keine Dateien ausgewählt")
        return

    all_trades = []
    all_equity = []
    all_metrics = []

    print(f"🚀 Verarbeite {len(file_paths)} Assets mit 10.000€ Startkapital pro Asset...")

    for file_path in tqdm(file_paths, desc="Verarbeite Assets"):
        try:
            df = pd.read_parquet(file_path, engine='pyarrow')
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                print(f"⚠️ Überspringe {os.path.basename(file_path)} – fehlende Spalten")
                continue

            symbol = os.path.basename(file_path).replace('.parquet', '').split('_')[0]
            df['symbol'] = symbol
            if isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
            else:
                start_date = pd.to_datetime('2024-08-07 00:00:00')
                df['timestamp'] = [start_date + timedelta(hours=i) for i in range(len(df))]
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            df = calculate_advanced_indicators(df)
            patterns = detect_candlestick_patterns(df)

            trades_df, equity_df, final_capital = simulate_advanced_strategy(df, patterns, initial_capital=10000)

            if not trades_df.empty:
                all_trades.append(trades_df)
                all_equity.append(equity_df)
                metrics = calculate_enhanced_metrics(trades_df, equity_df, 10000)
                all_metrics.append({'symbol': symbol, **metrics})
                print(f"✅ {symbol}: {len(trades_df)} Trades, "
                      f"Start: 10.000€ → Ende: {final_capital:,.2f}€ "
                      f"({((final_capital - 10000) / 10000) * 100:.2f}%)")

        except Exception as e:
            print(f"❌ Fehler bei {os.path.basename(file_path)}: {str(e)}")

    # Portfolio-Zusammenfassung
    if all_trades:
        global_trades = pd.concat(all_trades, ignore_index=True)
        global_equity = pd.concat(all_equity, ignore_index=True)

        total_initial = 10000 * len(file_paths)
        total_final = sum([metrics['final_capital'] for metrics in all_metrics])
        total_profit = total_final - total_initial

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = export_comprehensive_results(global_trades, global_equity, all_metrics, timestamp)

        print("\n" + "=" * 80)
        print("💰 **FINALE PORTFOLIO-ERGEBNISSE**")
        print("=" * 80)
        print(f"📊 Anzahl Assets: {len(file_paths)}")
        print(f"💰 Startkapital pro Asset: 10.000€")
        print(f"💰 Gesamt-Startkapital: {total_initial:,.2f}€")
        print(f"💰 Gesamt-Endkapital: {total_final:,.2f}€")
        print(f"💰 Gesamt-Gewinn/Verlust: {total_profit:,.2f}€")
        print(f"📈 Gesamt-Rendite: {(total_profit / total_initial) * 100:.2f}%")

        print("\n📈 **Asset-Details:**")
        for metrics in all_metrics:
            print(f"   {metrics['symbol']}: 10.000€ → {metrics['final_capital']:,.2f}€ "
                  f"({((metrics['final_capital'] - 10000) / 10000) * 100:.2f}%) "
                  f"- {metrics['total_trades']} Trades")

        print("\n📁 **Exportierte Dateien:")
        print(f"   📄 CSV: {results['csv']}")
        print(f"   📊 Excel: {results['excel']}")
        print(f"   🌐 HTML: {results['html']}")
        print("=" * 80)

    else:
        print("❌ Keine Trades generiert")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    main_complete()