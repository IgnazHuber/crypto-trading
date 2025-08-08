# kpi_analyzer_250805a.py
SKRIPT_ID = "kpi_analyzer_250805a"
"""
KPI-Auswertung: Sharpe, MaxDrawdown, CAGR, Winrate etc.
"""

import numpy as np
import pandas as pd

def calc_kpis(trades_df, start_capital=10000, freq='1h'):
    if trades_df.empty:
        return {
            "Trades": 0, "WinRate": 0, "TotalPnL_abs": 0, "TotalPnL_pct": 0,
            "AvgPnL%": 0, "MaxWin%": 0, "MaxLoss%": 0,
            "Sharpe": 0, "MaxDrawdown": 0, "CAGR": 0, "EndCapital": start_capital
        }
    trades_df = trades_df.sort_values('Exit Time')
    pnl = trades_df['PnL_abs']
    pnl_pct = trades_df['PnL_pct']
    capitals = trades_df['Kapital nach Trade']
    returns = capitals.pct_change().fillna(0)
    # Sharpe
    sharpe = np.sqrt(365*24) * returns.mean() / (returns.std()+1e-9)  # annualized, 1h->year
    # Drawdown
    running_max = capitals.cummax()
    drawdown = (capitals - running_max) / running_max
    max_drawdown = drawdown.min()
    # CAGR
    duration_years = (pd.to_datetime(trades_df['Exit Time']).max() - pd.to_datetime(trades_df['Entry Time']).min()).total_seconds() / (365*24*3600)
    cagr = (capitals.iloc[-1] / start_capital) ** (1/duration_years) - 1 if duration_years > 0 else 0
    win_rate = (pnl > 0).mean()*100
    return {
        "Trades": len(trades_df),
        "WinRate": round(win_rate, 1),
        "TotalPnL_abs": round(pnl.sum(), 1),
        "TotalPnL_pct": round((capitals.iloc[-1] - start_capital) / start_capital * 100, 1),
        "AvgPnL%": round(pnl_pct.mean(), 1),
        "MaxWin%": round(pnl_pct.max(), 1),
        "MaxLoss%": round(pnl_pct.min(), 1),
        "Sharpe": round(sharpe, 2),
        "MaxDrawdown": round(max_drawdown*100, 2),
        "CAGR": round(cagr*100, 2),
        "EndCapital": round(capitals.iloc[-1], 1)
    }
