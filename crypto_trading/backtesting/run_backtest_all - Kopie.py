import os
import pandas as pd
import numpy as np 
import vectorbt as vbt
from crypto_trading.strategy.trend_macd_adx_volume import trend_signals
from crypto_trading.visualization.charts import plot_trades

# Verzeichnisse
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

symbols = ["BTC_USD", "ETH_USD", "BNB_USD", "SOL_USD"]
intervals = ["1d", "4h", "1h", "15m"]

def backtest_symbol(symbol, interval):
    data_path = os.path.join(DATA_DIR, symbol, f"{interval}.parquet")

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Fehlende Daten: {data_path}")

    # Daten laden
    df = pd.read_parquet(data_path)

    # Erwartete Spalten prüfen
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' fehlt in {data_path}")

    # Signale berechnen
    entries, exits, short_entries, short_exits = trend_signals(df, long_only=False)

    # Portfolio
    pf = vbt.Portfolio.from_signals(
        close=df["Close"],
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        size=np.inf,
        fees=0.001
    )

    trades = pf.trades.records_readable
    chart_path = os.path.join(RESULT_DIR, f"{symbol}_{interval}.png")
    plot_trades(df, entries, exits, short_entries, short_exits, chart_path)

    return pf, trades, chart_path

def main():
    results = []

    for sym in symbols:
        for interval in intervals:
            print(f"Backtest {sym} {interval}\n")
            pf, trades, chart_path = backtest_symbol(sym, interval)

            stats = pf.stats()

            results.append({
                "symbol": sym,
                "interval": interval,
                "total_return": stats["Total Return [%]"],
                "sharpe_ratio": stats["Sharpe Ratio"],
                "max_drawdown": stats["Max Drawdown [%]"],
                "trades": len(trades),
                "chart": chart_path
            })

    result_df = pd.DataFrame(results)

    # **Fix: Zeitzonen entfernen**
    for col in result_df.select_dtypes(include=["datetimetz"]).columns:
        result_df[col] = result_df[col].dt.tz_localize(None)

    result_path = os.path.join(RESULT_DIR, "backtest_summary.xlsx")
    print(f"Speichere Ergebnisse: {result_path}")
    result_df.to_excel(result_path, index=False)

if __name__ == "__main__":
    main()
