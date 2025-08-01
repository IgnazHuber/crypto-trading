# backtesting/engine.py
import numpy as np

class BacktestEngine:
    def __init__(self, trades_df):
        self.trades = trades_df.copy()
        self.trades["pnl_abs"] = self.trades["exit_price"] - self.trades["entry_price"]
        self.trades["pnl_pct"] = (self.trades["pnl_abs"] / self.trades["entry_price"]) * 100

    def summary(self):
        total_pnl = self.trades["pnl_abs"].sum()
        win_rate = (self.trades["pnl_abs"] > 0).mean() * 100
        return {"TotalPnL": total_pnl, "WinRate": f"{win_rate:.2f}%"}
