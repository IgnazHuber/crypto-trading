import pandas as pd
import numpy as np

def calculate_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame, initial_capital: float):
    """Grundlegende Performance-Kennzahlen."""
    if trades_df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'final_capital': initial_capital,
            'return_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'sortino_ratio': 0,
            'omega_ratio': 0
        }

    metrics = {}
    metrics['total_trades'] = len(trades_df)
    metrics['win_rate'] = len(trades_df[trades_df['profit'] > 0]) / len(trades_df)
    metrics['total_profit'] = trades_df['profit'].sum()
    metrics['final_capital'] = initial_capital + metrics['total_profit']
    metrics['return_pct'] = (metrics['final_capital'] - initial_capital) / initial_capital * 100

    if len(equity_df) > 1:
        equity_df = equity_df.copy()
        equity_df['returns'] = equity_df['capital'].pct_change().fillna(0)
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(equity_df['returns'])
        metrics['max_drawdown'] = calculate_max_drawdown(equity_df['capital'])
        metrics['sortino_ratio'] = calculate_sortino_ratio(equity_df['returns'])
        metrics['omega_ratio'] = calculate_omega_ratio(equity_df['returns'])
    else:
        metrics['sharpe_ratio'] = 0
        metrics['max_drawdown'] = 0
        metrics['sortino_ratio'] = 0
        metrics['omega_ratio'] = 0

    return metrics

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate=0):
    """Annualisierte Sharpe Ratio."""
    excess_returns = returns - risk_free_rate / 252
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()
    if std_return == 0:
        return 0
    return (mean_return / std_return) * np.sqrt(252)

def calculate_max_drawdown(capital_series: pd.Series):
    """Maximaler Drawdown in Prozent."""
    running_max = capital_series.cummax()
    drawdowns = (running_max - capital_series) / running_max
    return drawdowns.max() * 100 if not drawdowns.empty else 0

def calculate_sortino_ratio(returns: pd.Series, required_return=0):
    """Annualisierte Sortino Ratio (beachtet nur Downside Risiko)."""
    downside_returns = returns[returns < required_return]
    downside_std = downside_returns.std()
    mean_return = returns.mean()
    if downside_std == 0:
        return 0
    return (mean_return - required_return) / downside_std * np.sqrt(252)

def calculate_omega_ratio(returns: pd.Series, threshold=0):
    """Berechnet die Omega Ratio (Verhältnis Gewinne zu Verlusten über Schwellenwert)."""
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    sum_gains = gains.sum()
    sum_losses = losses.sum()
    if sum_losses == 0:
        return np.inf if sum_gains > 0 else 0
    return sum_gains / sum_losses

def rolling_sharpe_ratio(equity_df: pd.DataFrame, window:int=60):
    """Gleitende Sharpe Ratio; window in Zeitperioden (z.B. Tage)."""
    equity_df = equity_df.copy()
    equity_df['returns'] = equity_df['capital'].pct_change().fillna(0)
    rolling_mean = equity_df['returns'].rolling(window).mean()
    rolling_std = equity_df['returns'].rolling(window).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
    return rolling_sharpe

def calculate_enhanced_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame, initial_capital: float):
    """Erweiterte Kennzahlen inkl. Trade-Level Auswertung und Risiko-Kennzahlen."""
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
            'profit_factor': abs(trades_df[trades_df['profit'] > 0]['profit'].sum() / trades_df[trades_df['profit'] < 0]['profit'].sum()) if trades_df[trades_df['profit'] < 0]['profit'].sum() != 0 else np.inf,
            'avg_duration': trades_df['duration_hours'].mean(),
            'calmar_ratio': metrics['return_pct'] / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else 0,
            'expectancy': calculate_expectancy(trades_df),
            'kelly_criterion': calculate_kelly_criterion(trades_df),
            'rolling_sharpe_avg': rolling_sharpe_ratio(equity_df).mean() if len(equity_df) > 0 else np.nan
        })

    return metrics

def calculate_expectancy(trades_df: pd.DataFrame):
    """Erwartungswert pro Trade."""
    if trades_df.empty:
        return 0
    win_rate = len(trades_df[trades_df['profit'] > 0]) / len(trades_df)
    avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean() or 0
    avg_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].mean()) or 0
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

def calculate_kelly_criterion(trades_df: pd.DataFrame):
    """Kelly-Kriterium zur optimalen Positionsgröße."""
    if trades_df.empty:
        return 0
    wins = trades_df[trades_df['profit'] > 0]['profit']
    losses = trades_df[trades_df['profit'] < 0]['profit']
    if wins.empty or losses.empty:
        return 0
    win_rate = len(wins) / len(trades_df)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    if avg_win == 0:
        return 0
    return (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

def trade_cluster_summary(trades_df: pd.DataFrame, bins: int = 3):
    if trades_df.empty:
        return pd.DataFrame()

    df = trades_df.copy()
    try:
        # Versuch mit Labels
        volatility_class = pd.qcut(
            df['size_eur'],
            q=bins,
            labels=[f'Gruppe_{i+1}' for i in range(bins)],
            duplicates='drop'
        )
    except ValueError:
        # Fallback ohne Labels (automatische Kategorien), dann in Strings konvertieren
        volatility_class = pd.qcut(
            df['size_eur'],
            q=bins,
            duplicates='drop'
        )
        volatility_class = volatility_class.astype(str)

    df['volatility_class'] = volatility_class

    cluster_summary = df.groupby(['regime', 'volatility_class', 'hour', 'type']).agg({
        'profit': ['count', 'mean', 'sum'],
        'return_pct': 'mean',
        'duration_hours': 'mean',
        'trade_nr': 'count'
    })

    cluster_summary.columns = ['_'.join(col) for col in cluster_summary.columns]
    cluster_summary = cluster_summary.reset_index()
    return cluster_summary



def monte_carlo_simulation(trades_df: pd.DataFrame, n_simulations: int=1000, initial_capital: float=10000):
    """
    Monte Carlo Simulation zur Visualisierung der möglichen Equity-Kurven aus historischen Returns.
    Gibt Wahrscheinlichkeitsverteilungen, Varianz und Drawdown-Quantile zurück.
    """
    if trades_df.empty or 'return_pct' not in trades_df.columns:
        return {}

    returns = trades_df['return_pct'].dropna().values / 100.0
    if len(returns) == 0:
        return {}

    simulations = []
    for _ in range(n_simulations):
        sim_returns = np.random.choice(returns, size=len(returns))
        equity = initial_capital
        equity_path = [equity]

        for r in sim_returns:
            equity *= (1 + r)
            equity_path.append(equity)

        equity_path = np.array(equity_path)
        running_max = np.maximum.accumulate(equity_path)
        drawdowns = (running_max - equity_path) / running_max
        max_drawdown = drawdowns.max() * 100

        simulations.append({'final_value': equity_path[-1], 'max_drawdown': max_drawdown})

    sim_df = pd.DataFrame(simulations)
    return {
        'expected_final_value': sim_df['final_value'].mean(),
        'median_final_value': sim_df['final_value'].median(),
        'var_95': sim_df['final_value'].quantile(0.05),
        'var_99': sim_df['final_value'].quantile(0.01),
        'max_drawdown_95': sim_df['max_drawdown'].quantile(0.95),
        'max_drawdown_99': sim_df['max_drawdown'].quantile(0.99)
    }

def max_drawdown_per_period(equity_df: pd.DataFrame, period: str = 'M'):
    """
    Max. Drawdown pro Zeitperiode (z.B. 'M' Monat, 'W' Woche).
    Gibt DataFrame mit Periodenstart und maximalem Drawdown zurück.
    """
    if equity_df.empty:
        return pd.DataFrame()

    equity_df = equity_df.copy()
    equity_df['period'] = equity_df['timestamp'].dt.to_period(period)
    results = []

    for period_key, group in equity_df.groupby('period'):
        capital_series = group['capital']
        max_dd = calculate_max_drawdown(capital_series)
        results.append({'period': period_key.start_time, 'max_drawdown_pct': max_dd})

    return pd.DataFrame(results)
