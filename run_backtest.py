# run_backtest.py

from crypto_trading.config import ASSETS, DATA_DIR, TIME_RANGE
from crypto_trading.trades import load_asset_data, compute_indicators, generate_trades_for_symbol
import pandas as pd

indicator_weights = {"MACD": 1, "RSI": 1, "ADX": 1, "BBANDS": 1}  # Für Score-Bildung, beliebig erweiterbar

all_trades = []
for symbol in ASSETS:
    try:
        df = load_asset_data(symbol, data_dir=DATA_DIR)
        df = df.last("365D") if hasattr(df, "last") else df[-365:]
        df = compute_indicators(df)
        trades = generate_trades_for_symbol(
            df, symbol, entry_threshold=3, exit_threshold=1, indicator_weights=indicator_weights
        )
        print(f"{symbol}: {len(trades)} Trades erzeugt")
        all_trades.extend(trades)
    except Exception as e:
        print(f"Fehler bei {symbol}: {e}")

trades_df = pd.DataFrame(all_trades)
print(trades_df[["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "pnl_abs", "pnl_pct"]].head())
print("Total Trades:", len(trades_df))

