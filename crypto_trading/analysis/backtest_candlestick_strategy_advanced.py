import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import os
from itertools import product
from collections import defaultdict

# ========== EINSTELLUNGEN ==========
DATA_PATH = r"c:\Projekte\crypto_trading\crypto_trading\data\raw"
DEFAULT_FILE = "BTCUSDT_1h_1year_ccxt.parquet"
EXPORT_CSV = True
EXPORT_STRATEGY_DOC = True
EXPORT_PERF_CSV = True

# === Parameter-Grid (beispielhaft!) ===
GRID = {
    "RSI_PERIOD": [14],
    "MACD_FAST": [12],
    "MACD_SLOW": [26],
    "MACD_SIGNAL": [9],
    "EMA_SHORT": [50],
    "EMA_LONG": [200],
    "BB_WINDOW": [20],
    "STOCH_WINDOW": [14],
    "STOP_LOSS_PCT": [0.03, 0.05],
    "TAKE_PROFIT_PCT": [0.06, 0.1],
}

# ========== Parquet-Auswahl/BATCH ==========
files = [f for f in os.listdir(DATA_PATH) if f.endswith('.parquet')]
print("Verfügbare Parquet-Dateien:")
for idx, fname in enumerate(files):
    print(f"{idx+1:2d}. {fname}")
multi_choice = input(f"Batch-Modus für mehrere Dateien? [z.B. 1,2,5 oder Enter für Default ({DEFAULT_FILE})]: ")
if multi_choice.strip():
    selection = [int(i.strip())-1 for i in multi_choice.split(',') if i.strip().isdigit()]
    parquet_files = [os.path.join(DATA_PATH, files[i]) for i in selection if 0 <= i < len(files)]
else:
    parquet_files = [os.path.join(DATA_PATH, DEFAULT_FILE)]

# === Grid-Search-Iterationen ===
param_grid = list(product(*GRID.values()))
param_names = list(GRID.keys())

# ======= Ergebnis-Container =======
all_results = []

# ========== Hilfsfunktionen ==========

def run_strategy(df, params, side='long'):
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, EMA_SHORT, EMA_LONG, BB_WINDOW, STOCH_WINDOW, STOP_LOSS_PCT, TAKE_PROFIT_PCT = params
    # Indikatoren berechnen
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=RSI_PERIOD).rsi()
    macd = ta.trend.MACD(df['close'], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema_short'] = ta.trend.EMAIndicator(df['close'], window=EMA_SHORT).ema_indicator()
    df['ema_long'] = ta.trend.EMAIndicator(df['close'], window=EMA_LONG).ema_indicator()
    bb = ta.volatility.BollingerBands(df['close'], window=BB_WINDOW)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=STOCH_WINDOW)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df = df.dropna()
    # Logik
    entries = []
    position = None
    entry_price = 0
    for i, row in df.iterrows():
        if side == 'long':
            entry = (
                row['rsi'] < 30 and
                row['macd'] > row['macd_signal'] and
                row['ema_short'] > row['ema_long'] and
                row['close'] <= row['bb_lower'] and
                row['stoch_k'] < 20 and row['stoch_k'] > row['stoch_d']
            )
            exit_ = (
                row['rsi'] > 60 or
                row['macd'] < row['macd_signal'] or
                row['close'] >= row['bb_upper'] or
                row['stoch_k'] > 80 or
                (row['close'] < entry_price * (1-STOP_LOSS_PCT)) or
                (row['close'] > entry_price * (1+TAKE_PROFIT_PCT))
            )
        elif side == 'short':
            entry = (
                row['rsi'] > 70 and
                row['macd'] < row['macd_signal'] and
                row['ema_short'] < row['ema_long'] and
                row['close'] >= row['bb_upper'] and
                row['stoch_k'] > 80 and row['stoch_k'] < row['stoch_d']
            )
            exit_ = (
                row['rsi'] < 40 or
                row['macd'] > row['macd_signal'] or
                row['close'] <= row['bb_lower'] or
                row['stoch_k'] < 20 or
                (row['close'] > entry_price * (1+STOP_LOSS_PCT)) or
                (row['close'] < entry_price * (1-TAKE_PROFIT_PCT))
            )
        else:
            raise ValueError("side muss 'long' oder 'short' sein")
        if position == side and exit_:
            entries[-1]['exit_time'] = i
            entries[-1]['exit_price'] = row['close']
            entries[-1]['pnl_pct'] = ((row['close'] - entry_price)/entry_price * 100) if side == 'long' else ((entry_price - row['close'])/entry_price * 100)
            position = None
        if position is None and entry:
            entries.append({'entry_time': i, 'entry_price': row['close']})
            entry_price = row['close']
            position = side
    # Offene Position schließen
    if position == side and entries:
        last = df.iloc[-1]
        entries[-1]['exit_time'] = last.name
        entries[-1]['exit_price'] = last['close']
        entries[-1]['pnl_pct'] = ((last['close'] - entry_price)/entry_price * 100) if side == 'long' else ((entry_price - last['close'])/entry_price * 100)
    return pd.DataFrame(entries)

def performance_metrics(trades, df, params, asset, direction):
    # Equity Curve & Returns berechnen
    if trades.empty:
        return {
            "Asset": asset,
            "Direction": direction,
            **{k: v for k, v in zip(param_names, params)},
            "Anzahl Trades": 0, "Gesamt-PnL": np.nan, "Trefferquote": np.nan,
            "Ø Trade-PnL": np.nan, "Max. Gewinn": np.nan, "Max. Verlust": np.nan,
            "Sharpe": np.nan, "MaxDrawdown": np.nan, "Letztes Kapital": np.nan
        }
    trades = trades.copy()
    trades['holding_period'] = (trades['exit_time'] - trades['entry_time']).astype('timedelta64[h]')
    trades['direction'] = direction
    total_pnl = trades['pnl_pct'].sum()
    num_trades = len(trades)
    win_rate = (trades['pnl_pct'] > 0).mean() * 100
    avg_pnl = trades['pnl_pct'].mean()
    max_gain = trades['pnl_pct'].max()
    max_loss = trades['pnl_pct'].min()
    # Equity Curve
    equity = pd.Series(index=df.index, dtype=float)
    equity[:] = 1.0  # Startkapital = 1
    for idx, trade in trades.iterrows():
        entry, exit = trade['entry_time'], trade['exit_time']
        pnl = trade['pnl_pct'] / 100.0
        if pd.notna(entry) and pd.notna(exit):
            equity.loc[exit:] = equity.loc[exit:] * (1 + pnl)
    equity = equity.fillna(method='ffill')
    returns = equity.pct_change().dropna()
    # Sharpe (auf annualisierte Stundenbasis, risikofrei 0)
    sharpe = returns.mean() / returns.std() * np.sqrt(24*365) if returns.std() > 0 else np.nan
    # Max Drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown = drawdown.min()
    # Letztes Kapital
    last_equity = equity.dropna().iloc[-1] if not equity.dropna().empty else np.nan
    # Monatsauswertung
    trades['entry_month'] = pd.to_datetime(trades['entry_time']).dt.to_period('M')
    monthly = trades.groupby('entry_month')['pnl_pct'].sum()
    # Resultat
    result = {
        "Asset": asset,
        "Direction": direction,
        **{k: v for k, v in zip(param_names, params)},
        "Anzahl Trades": num_trades, "Gesamt-PnL": total_pnl, "Trefferquote": win_rate,
        "Ø Trade-PnL": avg_pnl, "Max. Gewinn": max_gain, "Max. Verlust": max_loss,
        "Sharpe": sharpe, "MaxDrawdown": max_drawdown, "Letztes Kapital": last_equity,
        "Monthly": monthly
    }
    return result

def plot_equity_curve(equity, asset, direction, params):
    plt.figure(figsize=(14, 5))
    plt.plot(equity.index, equity.values, label='Equity Curve')
    plt.title(f'Equity Curve - {asset} ({direction})\nParameter: {params}')
    plt.xlabel('Zeit')
    plt.ylabel('Kapital')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_monthly_table(monthly):
    if monthly is None or monthly.empty:
        print("Keine Monatswerte vorhanden.")
        return
    print("\nMonats-Performance:")
    print(monthly.to_frame("Sum PnL [%]"))

# ========== BATCH + GRID LOOP ==========
for parquet_file in parquet_files:
    df = pd.read_parquet(parquet_file)
    asset = os.path.basename(parquet_file).split("_")[0]
    df = df.sort_index()
    if not np.issubdtype(df.index.dtype, np.datetime64):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            df.set_index('timestamp', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=['close', 'high', 'low'])
    print(f"\nAnalysiere: {parquet_file}")

    for params in param_grid:
        for direction in ["long", "short"]:
            trades = run_strategy(df, params, side=direction)
            perf = performance_metrics(trades, df, params, asset, direction)
            all_results.append(perf)
            # Optional: Plot nur für eine Param-Kombination
            if (len(param_grid) == 1) and (direction == "long"):
                print("\n=== Trade-Auswertung ===")
                print(trades[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl_pct', 'holding_period']].head(10))
                print(f"\nGesamt-PnL: {perf['Gesamt-PnL']:.2f}%")
                print(f"Anzahl Trades: {perf['Anzahl Trades']}")
                print(f"Trefferquote: {perf['Trefferquote']:.2f}%")
                print(f"Ø Trade-PnL: {perf['Ø Trade-PnL']:.2f}%")
                print(f"Max. Einzelgewinn: {perf['Max. Gewinn']:.2f}%")
                print(f"Max. Einzelverlust: {perf['Max. Verlust']:.2f}%")
                print(f"Sharpe Ratio: {perf['Sharpe']:.2f}")
                print(f"Max Drawdown: {perf['MaxDrawdown']:.2%}")
                print(f"Letztes Kapital: {perf['Letztes Kapital']:.4f}")
                plot_monthly_table(perf["Monthly"])
                # Equity Curve plot
                # Equity Curve rekonstruieren (analog performance_metrics)
                equity = pd.Series(index=df.index, dtype=float)
                equity[:] = 1.0
                for idx, trade in trades.iterrows():
                    entry, exit = trade['entry_time'], trade['exit_time']
                    pnl = trade['pnl_pct'] / 100.0
                    if pd.notna(entry) and pd.notna(exit):
                        equity.loc[exit:] = equity.loc[exit:] * (1 + pnl)
                equity = equity.fillna(method='ffill')
                plot_equity_curve(equity, asset, direction, dict(zip(param_names, params)))

# ========== Export aller Ergebnisse ==========
df_result = pd.DataFrame([{k: v for k, v in perf.items() if k != 'Monthly'} for perf in all_results])
if EXPORT_PERF_CSV:
    df_result.to_csv("strategy_performance_grid.csv", index=False)
    print(f"\nPerformance-Grid gespeichert: strategy_performance_grid.csv")

print("\nFertig. Übersicht siehe strategy_performance_grid.csv")
