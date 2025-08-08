import concurrent.futures
import pandas as pd
import os

from .strategy import simulate_advanced_strategy
from . import metrics

# Import detect_candlestick_patterns von crypto_trading.analysis
from crypto_trading.analysis.candlestick_analyzer import detect_candlestick_patterns


def run_single_backtest(file_path, initial_capital=10000):
    import pyarrow.parquet as pq

    df = pd.read_parquet(file_path, engine='pyarrow')

    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns in {os.path.basename(file_path)}")

    symbol = os.path.basename(file_path).replace('.parquet', '').split('_')[0]
    df['symbol'] = symbol

    if isinstance(df.index, pd.DatetimeIndex):
        df['timestamp'] = df.index
    else:
        start_date = pd.to_datetime('2024-08-07 00:00:00')
        df['timestamp'] = [start_date + pd.Timedelta(hours=i) for i in range(len(df))]

    # korrekter relativer Import der Indikatorenfunktion
    from .indicators import calculate_advanced_indicators

    df = calculate_advanced_indicators(df)
    patterns = detect_candlestick_patterns(df)

    trades_df, equity_df, final_capital = simulate_advanced_strategy(
        df, patterns, initial_capital=initial_capital)

    return trades_df, equity_df, final_capital, symbol


def run_backtests_parallel(file_paths, initial_capital=10000, max_workers=None):
    all_trades = []
    all_equity = []
    all_metrics = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_backtest, fp, initial_capital) for fp in file_paths]
        for future in concurrent.futures.as_completed(futures):
            try:
                trades_df, equity_df, final_capital, symbol = future.result()
                if not trades_df.empty:
                    all_trades.append(trades_df)
                    all_equity.append(equity_df)
                    metric = metrics.calculate_metrics(trades_df, equity_df, initial_capital)
                    metric['symbol'] = symbol
                    all_metrics.append(metric)
                print(f"✅ {symbol}: {len(trades_df)} Trades, Endkapital: {final_capital:.2f}€")
            except Exception as e:
                print(f"❌ Fehler in Backtest: {str(e)}")

    return all_trades, all_equity, all_metrics
