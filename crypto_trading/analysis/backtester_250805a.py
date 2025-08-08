# backtester.py
SKRIPT_ID = "backtester_250805a"
"""
Backtesting-Logik für das Krypto-Framework: Generiert Trades nach Entry/Exit-Signalen.
"""

import pandas as pd

def run_strategy(df, candle_func, regime_mask, strat_name, regime_name, direction="long", start_capital=10000, asset="BTCUSDT", min_capital=100):
    trades = []
    capital = start_capital
    in_position, entry_price, entry_idx, entry_time = False, None, None, None
    trade_id = 1
    for idx in df.index:
        if capital < min_capital:
            break
        entry_signal = candle_func(df).loc[idx] and regime_mask.loc[idx]
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
                direction_mult = 1 if direction == "long" else -1
                pnl = direction_mult * (exit_price - entry_price) / entry_price * einsatz
                pnl = max(-0.05 * einsatz, min(pnl, 0.20 * einsatz))
                capital += pnl
                trades.append({
                    "Trade-ID": int(trade_id),
                    "Asset": asset,
                    "Strategy": strat_name,
                    "Regime": regime_name,
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

if __name__ == "__main__":
    print(f"[{SKRIPT_ID}] Test: Dummy-Backtest")
    # Dummy-Daten zum Testen (hier weiter ausbauen)
