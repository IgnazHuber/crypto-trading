import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import os
from itertools import product

# ========== EINSTELLUNGEN ==========
DATA_PATH = r"c:\Projekte\crypto_trading\crypto_trading\data\raw"
DEFAULT_FILES = ["BTCUSDT_1h_1year_ccxt.parquet"]  # <- hier beliebig viele Defaults (auch ETHUSDT, etc.)
EXPORT_TRADE_CSV = True
EXPORT_PERF_CSV = True

# Score-basierte Entry: ab wie vielen Kriterien wird gekauft? (z. B. 3 von 5)
SCORE_ENTRY_THRESHOLD = 3

# ==== Parameter-Grid (beliebig anpassen/erweitern!) ====
GRID = {
    "RSI_PERIOD": [8, 14, 21],
    "MACD_FAST": [8, 12],
    "MACD_SLOW": [18, 26],
    "MACD_SIGNAL": [6, 9],
    "EMA_SHORT": [21, 50],
    "EMA_LONG": [100, 200],
    "BB_WINDOW": [10, 20],
    "STOCH_WINDOW": [8, 14],
    "STOP_LOSS_PCT": [0.02, 0.03, 0.05],
    "TAKE_PROFIT_PCT": [0.04, 0.06, 0.1],
}
param_grid = list(product(*GRID.values()))
param_names = list(GRID.keys())

# ========== Datei-Auswahl ==========
files = [f for f in os.listdir(DATA_PATH) if f.endswith('.parquet')]
print("Verfügbare Parquet-Dateien:")
for idx, fname in enumerate(files):
    print(f"{idx+1:2d}. {fname}")
multi_choice = input(f"Mehrere Dateien im Batch? (z.B. 1,2,5 oder Enter für Defaults ({', '.join(DEFAULT_FILES)})): ")
if multi_choice.strip():
    selection = [int(i.strip())-1 for i in multi_choice.split(',') if i.strip().isdigit()]
    parquet_files = [os.path.join(DATA_PATH, files[i]) for i in selection if 0 <= i < len(files)]
else:
    parquet_files = [os.path.join(DATA_PATH, f) for f in DEFAULT_FILES]

# ======= Ergebnis-Container =======
all_results = []
all_trades = []

# ========== STRATEGIE ==========

def run_score_strategy(df, params, asset, threshold=SCORE_ENTRY_THRESHOLD):
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, EMA_SHORT, EMA_LONG, BB_WINDOW, STOCH_WINDOW, STOP_LOSS_PCT, TAKE_PROFIT_PCT = params
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

    entries = []
    position = None
    entry_price = 0
    entry_score = 0
    entry_reasons = ""
    for i, row in df.iterrows():
        # Score-Logik (jeder Treffer 1 Punkt)
        score = 0
        reasons = []
        if row['rsi'] < 35:
            score += 1
            reasons.append("RSI < 35")
        if row['macd'] > row['macd_signal']:
            score += 1
            reasons.append("MACD Bull-Crossover")
        if row['ema_short'] > row['ema_long']:
            score += 1
            reasons.append(f"EMA({EMA_SHORT}) > EMA({EMA_LONG})")
        if row['close'] <= row['bb_lower']:
            score += 1
            reasons.append("Kurs <= BB lower")
        if row['stoch_k'] < 25 and row['stoch_k'] > row['stoch_d']:
            score += 1
            reasons.append("Stoch %K < 25 & > %D")
        # Entry ab Threshold
        entry = (score >= threshold)
        exit_ = (
            (row['rsi'] > 60) or
            (row['macd'] < row['macd_signal']) or
            (row['close'] >= row['bb_upper']) or
            (row['stoch_k'] > 80) or
            (row['close'] < entry_price * (1 - STOP_LOSS_PCT)) or
            (row['close'] > entry_price * (1 + TAKE_PROFIT_PCT))
        )
        if position and exit_:
            pnl_pct = (row['close'] - entry_price) / entry_price * 100
            reason = []
            if row['rsi'] > 60: reason.append("RSI > 60")
            if row['macd'] < row['macd_signal']: reason.append("MACD Bear-Crossover")
