# portfolio_metrics.py
import numpy as np
import pandas as pd

def calculate_sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    returns = equity_curve.pct_change().dropna()
    excess_returns = returns - risk_free_rate / len(returns)
    return np.sqrt(252) * excess_returns.mean() / returns.std() if not returns.empty else np.nan

def calculate_sortino_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    returns = equity_curve.pct_change().dropna()
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if not downside_returns.empty else np.nan
    return np.sqrt(252) * (returns.mean() - risk_free_rate / len(returns)) / downside_std if downside_std and not np.isnan(downside_std) else np.nan

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1.0
    return drawdown.min() if not drawdown.empty else np.nan

def calculate_cagr(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return np.nan
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    num_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    return (end_value / start_value) ** (1 / num_years) - 1 if num_years > 0 else np.nan

def calculate_volatility(equity_curve: pd.Series) -> float:
    returns = equity_curve.pct_change().dropna()
    return np.sqrt(252) * returns.std() if not returns.empty else np.nan

def calculate_portfolio_metrics(equity_curves: dict) -> pd.DataFrame:
    results = []
    for name, curve in equity_curves.items():
        results.append({
            "Asset": name,
            "Sharpe": calculate_sharpe_ratio(curve),
            "Sortino": calculate_sortino_ratio(curve),
            "Volatility": calculate_volatility(curve),
            "MaxDrawdown": calculate_max_drawdown(curve),
            "CAGR": calculate_cagr(curve)
        })
    return pd.DataFrame(results)
