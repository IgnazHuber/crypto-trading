import os
import numpy as np
import pandas as pd
import vectorbt as vbt
from tqdm import tqdm
from crypto_trading.visualization.charts import plot_trades

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

def trend_signals(df, long_only=False, **kwargs):
    n = len(df)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    short_entries = np.zeros(n, dtype=bool)
    short_exits = np.zeros(n, dtype=bool)
    return entries, exits, short_entries, short_exits

def backtest_symbol(symbol, interval, max_rows=None):
    data_path = os.path.join(DATA_DIR, symbol, f"{interval}.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fehlende Daten: {data_path}")

    df = pd.read_parquet(data_path)
    if max_rows:
        df = df.tail(max_rows)

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

def main(symbols=None, intervals=None, max_rows=None):
    # Ermittle Testmodus aus ENV (wird via conftest/pytest gesetzt)
    if symbols is None or intervals is None or max_rows is None:
        TESTMODE = os.environ.get("CRYPTO_TESTMODE", "0") == "1"
        TESTFAST = os.environ.get("CRYPTO_TESTFAST", "0") == "1"
        if TESTMODE and TESTFAST:
            symbols = ["BTC_USD"]
            intervals = ["1d"]
            max_rows = 10
        elif TESTMODE:
            symbols = ["BTC_USD"]
            intervals = ["1d"]
            max_rows = 100
        else:
            symbols = ["BTC_USD", "ETH_USD", "BNB_USD", "SOL_USD"]
            intervals = ["1d", "4h", "1h", "15m"]
            max_rows = None

    result_rows = []
    for sym in tqdm(symbols, desc="Assets"):
        for interval in tqdm(intervals, desc=f"{sym} Intervals", leave=False):
            try:
                pf, trades, chart_path = backtest_symbol(sym, interval, max_rows=max_rows)
                stats = pf.stats()
                stats["symbol"] = sym
                stats["interval"] = interval
                stats["chart"] = chart_path
                result_rows.append(stats)
            except Exception as e:
                print(f"Fehler bei {sym} {interval}: {e}")

    result_df = pd.DataFrame(result_rows)
    for col in result_df.select_dtypes(include=["datetimetz"]).columns:
        result_df[col] = result_df[col].dt.tz_convert(None)

    result_path = os.path.join(RESULT_DIR, "backtest_summary.xlsx")
    print(f"Speichere Ergebnisse: {result_path}")
    result_df.to_excel(result_path, index=False)
    print("Fertig.")

# Für direkte Aufrufe:
if __name__ == "__main__":
    main()
