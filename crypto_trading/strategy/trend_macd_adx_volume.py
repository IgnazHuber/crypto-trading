import os
import numpy as np
import pandas as pd
import vectorbt as vbt
from crypto_trading.visualization.charts import plot_trades

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

SYMBOLS = ["BTC_USD", "ETH_USD", "BNB_USD", "SOL_USD"]
INTERVALS = ["1d", "4h", "1h", "15m"]

import os

TESTMODE = os.environ.get("CRYPTO_TESTMODE", "0") == "1"
TESTFAST = os.environ.get("CRYPTO_TESTFAST", "0") == "1"

if TESTMODE and TESTFAST:
    # Sehr schneller Test: nur 1 Asset, 1 Intervall, 10 Zeilen!
    SYMBOLS = ["BTC_USD"]
    INTERVALS = ["1d"]
    MAX_ROWS = 10
elif TESTMODE:
    # Schneller Test: nur 1 Asset, 1 Intervall, 100 Zeilen!
    SYMBOLS = ["BTC_USD"]
    INTERVALS = ["1d"]
    MAX_ROWS = 100
else:
    # Volltest
    SYMBOLS = ["BTC_USD", "ETH_USD", "BNB_USD", "SOL_USD"]
    INTERVALS = ["1d", "4h", "1h", "15m"]
    MAX_ROWS = None


def trend_signals(df, long_only=False, **kwargs):
    """
    Dummy: Liefert vier leere Signalarays, akzeptiert Argument 'long_only'.
    """
    n = len(df)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    short_entries = np.zeros(n, dtype=bool)
    short_exits = np.zeros(n, dtype=bool)
    return entries, exits, short_entries, short_exits

def backtest_symbol(symbol, interval):
    data_path = os.path.join(DATA_DIR, symbol, f"{interval}.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fehlende Daten: {data_path}")

    df = pd.read_parquet(data_path)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    entries, exits, short_entries, short_exits = trend_signals(df, long_only=False)

    pf = vbt.Portfolio.from_signals(
        close=df["Close"],
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        size=1_000
    )

    trades = pf.trades.records_readable
    chart_path = os.path.join(RESULT_DIR, f"{symbol}_{interval}_trades.png")
    plot_trades(df, entries, exits, short_entries, short_exits, chart_path)

    return pf, trades, chart_path

def main():
    result_rows = []
    for sym in SYMBOLS:
        for interval in INTERVALS:
            print(f"Backtest {sym} {interval}")
            pf, trades, chart_path = backtest_symbol(sym, interval)
            stats = pf.stats()
            stats["symbol"] = sym
            stats["interval"] = interval
            stats["chart"] = chart_path
            result_rows.append(stats)

    result_df = pd.DataFrame(result_rows)
    for col in result_df.select_dtypes(include=["datetimetz"]).columns:
        result_df[col] = result_df[col].dt.tz_convert(None)

    result_path = os.path.join(RESULT_DIR, "backtest_summary.xlsx")
    print(f"Speichere Ergebnisse: {result_path}")
    result_df.to_excel(result_path, index=False)
    print("Fertig.")

if __name__ == "__main__":
    main()
