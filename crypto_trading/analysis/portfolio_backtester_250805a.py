# portfolio_backtester_250805a.py
SKRIPT_ID = "portfolio_backtester_250805a"
"""
Portfolio-/Meta-Backtest: Score-Signal als Handelslogik.
"""

import pandas as pd

def run_meta_strategy(df, signal_col, start_capital=10000, min_capital=100, asset="BTCUSDT"):
    trades = []
    capital = start_capital
    in_position = False
    entry_price, entry_idx, entry_time = None, None, None
    trade_id = 1
    for idx in df.index:
        if capital < min_capital:
            break
        entry_signal = df.loc[idx, signal_col]
        if not in_position and entry_signal:
            einsatz = min(0.10 * capital, capital)
            if einsatz < 1:
                continue
            in_position = True
            entry_price = df.loc[idx, 'close']
            entry_idx = idx
            entry_time = df.loc[idx, 'timestamp'] if 'timestamp' in df else idx
        elif in_position:
            price_now = df.loc[idx, 'close']
            stop_loss = entry_price * 0.95
            take_profit = entry_price * 1.20
            time_exceeded = (df.index.get_loc(idx) - df.index.get_loc(entry_idx) > 30)
            hit_stop = (price_now <= stop_loss)
            hit_tp = (price_now >= take_profit)
            if hit_stop or hit_tp or time_exceeded:
                exit_price = price_now
                exit_time = df.loc[idx, 'timestamp'] if 'timestamp' in df else idx
                pnl = (exit_price - entry_price) / entry_price * einsatz
                pnl = max(-0.05 * einsatz, min(pnl, 0.20 * einsatz))
                capital += pnl
                trades.append({
                    "Trade-ID": int(trade_id),
                    "Asset": asset,
                    "Strategy": f"Meta_Score_{signal_col}",
                    "Entry Time": entry_time,
                    "Exit Time": exit_time,
                    "Entry Price": round(entry_price, 1),
                    "Exit Price": round(exit_price, 1),
                    "Einsatz": round(einsatz, 1),
                    "PnL_abs": round(pnl, 1),
                    "PnL_pct": round(pnl / einsatz * 100, 1),
                    "Kapital nach Trade": round(capital, 1)
                })
                trade_id += 1
                in_position = False
    trades_df = pd.DataFrame(trades)
    return trades_df
