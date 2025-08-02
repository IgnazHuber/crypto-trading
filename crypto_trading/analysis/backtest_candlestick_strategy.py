import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import os

# ========== EINSTELLUNGEN ==========
DATA_PATH = r"c:\Projekte\crypto_trading\crypto_trading\data\raw"
DEFAULT_FILE = "BTCUSDT_1h_1year_ccxt.parquet"
EXPORT_CSV = True
EXPORT_STRATEGY_DOC = True

# Strategie-Parameter (für Feintuning)
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 50
EMA_LONG = 200
BB_WINDOW = 20
STOCH_WINDOW = 14
STOP_LOSS_PCT = 0.03
TAKE_PROFIT_PCT = 0.06

# ===== Strategie-Logik (wird unten auch dokumentiert!) =====
STRATEGY_DOC = f"""
Trading-Strategie auf Basis der 5 wichtigsten Candlestick-Indikatoren:
LONG:
- RSI < 30
- MACD-Linie > Signal-Linie (bullisches Crossover)
- EMA{EMA_SHORT} > EMA{EMA_LONG} (Trendfilter)
- Kurs <= unteres Bollinger Band
- Stochastik %K < 20 und %K > %D
Exit LONG:
- RSI > 60
- MACD < Signal
- Kurs >= oberes BB
- Stochastik > 80
- ODER Stop-Loss ({-STOP_LOSS_PCT*100:.1f}%) / Take-Profit (+{TAKE_PROFIT_PCT*100:.1f}%)
SHORT (optional):
- RSI > 70
- MACD-Linie < Signal-Linie (bearishes Crossover)
- EMA{EMA_SHORT} < EMA{EMA_LONG}
- Kurs >= oberes Bollinger Band
- Stochastik %K > 80 und %K < %D
Exit SHORT:
- RSI < 40
- MACD > Signal
- Kurs <= unteres BB
- Stochastik < 20
- ODER Stop-Loss / Take-Profit
"""

# ========== Parquet-Auswahl ==========
files = [f for f in os.listdir(DATA_PATH) if f.endswith('.parquet')]
print("Verfügbare Parquet-Dateien:")
for idx, fname in enumerate(files):
    print(f"{idx+1:2d}. {fname}")
try:
    selection = int(input(f"Datei auswählen [1-{len(files)}], Enter für Default ({DEFAULT_FILE}): ") or files.index(DEFAULT_FILE)+1)
except Exception:
    selection = files.index(DEFAULT_FILE)+1

parquet_file = os.path.join(DATA_PATH, files[selection-1])
print(f"\n--> Analyse von Datei: {parquet_file}")

# ========== Daten laden ==========
df = pd.read_parquet(parquet_file)
df = df.sort_index()
if not np.issubdtype(df.index.dtype, np.datetime64):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df.set_index('timestamp', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
df = df.dropna(subset=['close', 'high', 'low'])

# ========== Indikatoren berechnen ==========
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

# ========== Entry/Exit Logik ==========
def run_strategy(df, side='long'):
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

# ========== Strategieauswahl ==========
print("\nStrategie auswählen:")
print("1. Nur Long (klassisch, einfach zu interpretieren)")
print("2. Nur Short (nur zur Analyse, meist schwieriger in BTC-Longmarket)")
print("3. Long & Short abwechselnd (jede Richtung bei Signal)")
side_choice = input("Auswahl [1=Long, 2=Short, 3=Beides]: ") or "1"

if side_choice == "1":
    trades = run_strategy(df, side='long')
    strategy_desc = "Nur Long-Trades"
elif side_choice == "2":
    trades = run_strategy(df, side='short')
    strategy_desc = "Nur Short-Trades"
elif side_choice == "3":
    trades_long = run_strategy(df, side='long')
    trades_short = run_strategy(df, side='short')
    # Robust gegen leere DataFrames:
    if not trades_long.empty and not trades_short.empty:
        trades = pd.concat([trades_long.assign(direction='long'), trades_short.assign(direction='short')])
        trades = trades.sort_values('entry_time')
    elif not trades_long.empty:
        trades = trades_long.assign(direction='long')
    elif not trades_short.empty:
        trades = trades_short.assign(direction='short')
    else:
        trades = pd.DataFrame()
    trades.reset_index(drop=True, inplace=True)
    strategy_desc = "Long- und Short-Trades"
else:
    trades = run_strategy(df, side='long')
    strategy_desc = "Nur Long-Trades"

# ========== Performance-Auswertung ==========
if trades.empty:
    print("Keine Trades gefunden!")
else:
    trades['holding_period'] = (trades['exit_time'] - trades['entry_time']).astype('timedelta64[h]')
    total_pnl = trades['pnl_pct'].sum()
    num_trades = len(trades)
    win_rate = (trades['pnl_pct'] > 0).mean() * 100
    avg_pnl = trades['pnl_pct'].mean()
    max_loss = trades['pnl_pct'].min()
    max_gain = trades['pnl_pct'].max()
    trades['direction'] = trades.get('direction', 'long')

    print("\n=== Strategie-Logik ===")
    print(STRATEGY_DOC)
    print(f"Modus: {strategy_desc}")

    print("\n=== Trade-Auswertung ===")
    print(trades[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl_pct', 'holding_period', 'direction']])
    print(f"\nGesamt-PnL: {total_pnl:.2f}%")
    print(f"Anzahl Trades: {num_trades}")
    print(f"Trefferquote: {win_rate:.2f}%")
    print(f"Ø Trade-PnL: {avg_pnl:.2f}%")
    print(f"Max. Einzelgewinn: {max_gain:.2f}%")
    print(f"Max. Einzelverlust: {max_loss:.2f}%")

    # ========== Export ==========
    if EXPORT_CSV:
        trades.to_csv("trades_report.csv", index=False)
        print("\nTrade-Report als CSV gespeichert: trades_report.csv")
    if EXPORT_STRATEGY_DOC:
        with open("strategy_doc.txt", "w", encoding="utf-8") as f:
            f.write("=== Strategie-Logik ===\n")
            f.write(STRATEGY_DOC)
            f.write(f"\nModus: {strategy_desc}\n")
        print("Strategie-Doku gespeichert: strategy_doc.txt")

    # ========== Plot ==========
    plt.figure(figsize=(16, 7))
    plt.plot(df.index, df['close'], label="Kurs")
    if 'direction' in trades.columns:
        plt.scatter(trades[trades['direction']=='long']['entry_time'], trades[trades['direction']=='long']['entry_price'], marker='^', color='green', label='Long Entry')
        plt.scatter(trades[trades['direction']=='long']['exit_time'], trades[trades['direction']=='long']['exit_price'], marker='v', color='red', label='Long Exit')
        plt.scatter(trades[trades['direction']=='short']['entry_time'], trades[trades['direction']=='short']['entry_price'], marker='v', color='orange', label='Short Entry')
        plt.scatter(trades[trades['direction']=='short']['exit_time'], trades[trades['direction']=='short']['exit_price'], marker='^', color='blue', label='Short Exit')
    else:
        plt.scatter(trades['entry_time'], trades['entry_price'], marker='^', color='green', label='Entry')
        plt.scatter(trades['exit_time'], trades['exit_price'], marker='v', color='red', label='Exit')
    plt.title(f"{os.path.basename(parquet_file)} - {strategy_desc}")
    plt.xlabel("Zeit")
    plt.ylabel("Kurs")
    plt.legend()
    plt.tight_layout()
    plt.show()
