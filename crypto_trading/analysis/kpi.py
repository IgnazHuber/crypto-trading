"""
Modul: kpi.py

Berechnet wissenschaftliche KPIs für Einzelstrategien, Paramsets und Portfolios:
- Sharpe, Sortino, CAGR, Volatilität, Max/Mean Drawdown, Trefferquote, Rolling KPIs etc.
- Unterstützt Einzeltrades, Equity-Kurven und Multi-Asset-Portfolios

Funktionen:
- performance_metrics(trades, params, asset, param_names)
- extended_kpis(trades, eq_curve)
- rolling_sharpe(eq, window=500, freq_factor=24*365)
- rolling_sortino(eq, window=500, freq_factor=24*365)
- drawdown_series(eq)
- multi_asset_correlation(df_perf, metric="Gesamt-PnL")
- portfolio_kpis(df_all_trades, eq_curve=None)

Author: ChatGPT Research, 2025
"""

import pandas as pd
import numpy as np

def performance_metrics(trades, params, asset, param_names, freq_factor=24*365):
    """
    Liefert Basis-KPIs für ein Asset/Paramset.
    Args:
        trades: DataFrame mit Trades (einschl. Gewinn/Verlust)
        params: Paramset (tuple/list)
        asset: Assetname (z. B. BTC)
        param_names: Liste der Parameternamen (Grid)
        freq_factor: Annualisierungsfaktor (z. B. 24*365 für 1h-Daten)
    Returns:
        dict aller KPIs
    """
    if trades.empty:
        return {
            "Asset": asset,
            **{k: v for k, v in zip(param_names, params)},
            "Anzahl Trades": 0, "Gesamt-PnL": np.nan, "Trefferquote": np.nan,
            "Ø Trade-PnL": np.nan, "Max. Gewinn": np.nan, "Max. Verlust": np.nan,
            "Sharpe": np.nan, "MaxDrawdown": np.nan, "Letztes Kapital": np.nan,
            "Sortino": np.nan, "CAGR": np.nan, "Volatilität": np.nan, "MeanDrawdown": np.nan
        }
    trades = trades.copy()
    trades['holding_period'] = (pd.to_datetime(trades['exit_time']) - pd.to_datetime(trades['entry_time'])).dt.total_seconds() / 3600
    total_pnl = trades['gewinn_verlust'].sum()
    num_trades = len(trades)
    win_rate = (trades['gewinn_verlust'] > 0).mean() * 100
    avg_pnl = trades['gewinn_verlust'].mean()
    max_gain = trades['gewinn_verlust'].max()
    max_loss = trades['gewinn_verlust'].min()
    # Equity Curve
    eq = pd.Series(1.0, index=pd.to_datetime(trades['exit_time']))
    for idx, trade in trades.iterrows():
        pnl = trade['gewinn_verlust'] / 100.0
        eq.iloc[idx:] = eq.iloc[idx:] * (1 + pnl)
    returns = eq.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(freq_factor) if returns.std() > 0 else np.nan
    sortino = calc_sortino(returns, freq_factor)
    cagr = calc_cagr(eq)
    vol = returns.std() * np.sqrt(freq_factor)
    cummax = eq.cummax()
    drawdown = (eq - cummax) / cummax
    max_drawdown = drawdown.min()
    mean_drawdown = drawdown.mean()
    last_equity = eq.dropna().iloc[-1] if not eq.dropna().empty else np.nan
    return {
        "Asset": asset,
        **{k: v for k, v in zip(param_names, params)},
        "Anzahl Trades": num_trades, "Gesamt-PnL": total_pnl, "Trefferquote": win_rate,
        "Ø Trade-PnL": avg_pnl, "Max. Gewinn": max_gain, "Max. Verlust": max_loss,
        "Sharpe": sharpe, "MaxDrawdown": max_drawdown, "Letztes Kapital": last_equity,
        "Sortino": sortino, "CAGR": cagr, "Volatilität": vol, "MeanDrawdown": mean_drawdown
    }

def calc_cagr(eq):
    if len(eq) < 2: return np.nan
    days = (eq.index[-1] - eq.index[0]).days
    if days <= 0: return np.nan
    cagr = eq.iloc[-1] ** (365/days) - 1
    return cagr

def calc_sortino(returns, freq_factor):
    mean = returns.mean()
    neg_std = returns[returns < 0].std()
    if neg_std and neg_std > 0:
        return mean / neg_std * np.sqrt(freq_factor)
    else:
        return np.nan

def extended_kpis(trades, eq_curve, freq_factor=24*365, rolling_window=500):
    """
    Wissenschaftlich erweiterte KPIs (Rolling Sharpe/Sortino, Drawdown-Serie, etc.)
    """
    results = {}
    returns = eq_curve.pct_change().dropna()
    # Rolling Sharpe/Sortino (als Series)
    results['rolling_sharpe'] = rolling_sharpe(eq_curve, window=rolling_window, freq_factor=freq_factor)
    results['rolling_sortino'] = rolling_sortino(eq_curve, window=rolling_window, freq_factor=freq_factor)
    # Drawdown-Serie (als Series)
    results['drawdown_series'] = drawdown_series(eq_curve)
    return results

def rolling_sharpe(eq, window=500, freq_factor=24*365):
    returns = eq.pct_change().dropna()
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    sharpe = roll_mean / roll_std * np.sqrt(freq_factor)
    return sharpe

def rolling_sortino(eq, window=500, freq_factor=24*365):
    returns = eq.pct_change().dropna()
    roll_mean = returns.rolling(window).mean()
    roll_neg_std = returns.rolling(window).apply(lambda x: x[x<0].std(), raw=False)
    sortino = roll_mean / roll_neg_std * np.sqrt(freq_factor)
    return sortino

def drawdown_series(eq):
    cummax = eq.cummax()
    drawdown = (eq / cummax) - 1
    return drawdown

def multi_asset_correlation(df_perf, metric="Gesamt-PnL"):
    """
    Gibt die Korrelationsmatrix des gewünschten Performance-Metrics für alle Assets zurück.
    Args:
        df_perf: DataFrame mit Performance-Kennzahlen (multi-asset)
        metric: z. B. 'Gesamt-PnL' oder 'Sharpe'
    Returns:
        DataFrame Korrelationsmatrix
    """
    piv = df_perf.pivot(index='param_idx', columns='Asset', values=metric)
    return piv.corr()

def portfolio_kpis(df_all_trades, eq_curve=None):
    """
    Portfolio-Gesamtauswertung über alle Trades/Assets hinweg.
    Args:
        df_all_trades: Alle Einzeltrades
        eq_curve: Gesamtkapitalverlauf (optional, sonst aus Trades berechnet)
    Returns:
        Dict mit Portfolio-KPIs
    """
    einsatz = df_all_trades['kaufpreis'].sum()
    erloese = df_all_trades['verkaufspreis'].sum()
    gewinn_abs = erloese - einsatz
    gewinn_rel = (gewinn_abs / einsatz) * 100 if einsatz else 0
    kpis = {
        "Einsatz": einsatz, "Erlöse": erloese, "Gewinn (absolut)": gewinn_abs, "Gewinn (%)": gewinn_rel,
        "Trades": len(df_all_trades)
    }
    if eq_curve is not None:
        returns = eq_curve.pct_change().dropna()
        kpis.update({
            "Sharpe": returns.mean() / returns.std() * np.sqrt(24*365) if returns.std() > 0 else np.nan,
            "MaxDrawdown": drawdown_series(eq_curve).min(),
            "CAGR": calc_cagr(eq_curve)
        })
    return kpis

# Optionaler Self-Test
if __name__ == "__main__":
    print("kpi.py – Self-Test – bitte im Batch-Framework nutzen.")
