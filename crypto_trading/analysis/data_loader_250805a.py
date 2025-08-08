# data_loader_250805a.py
SKRIPT_ID = "data_loader_250805a"
"""
Datenimport und Indikatorberechnung für Krypto-Backtesting.
Neu: VWAP, MFI, Heikin Ashi, 4h-EMA für Multi-TF-Strategien.
"""

import pandas as pd
import numpy as np

def money_flow_index(df, period=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    pos_mf = pos_flow.rolling(period).sum()
    neg_mf = neg_flow.rolling(period).sum()
    mfi = 100 - 100 / (1 + (pos_mf / (neg_mf + 1e-9)))
    return mfi

def heikin_ashi(df):
    ha_df = pd.DataFrame(index=df.index)
    ha_df['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = []
    for i in range(len(df)):
        if i == 0:
            ha_open.append((df['open'].iloc[0] + df['close'].iloc[0]) / 2)
        else:
            ha_open.append((ha_open[-1] + ha_df['close'].iloc[i - 1]) / 2)
    ha_df['open'] = ha_open
    ha_df['high'] = pd.concat([df['high'], ha_df['open'], ha_df['close']], axis=1).max(axis=1)
    ha_df['low'] = pd.concat([df['low'], ha_df['open'], ha_df['close']], axis=1).min(axis=1)
    return ha_df

def add_ema4h(df):
    # Simpler Ansatz: 4h EMA = auf 1h-Chart rolling window (4) -> dann EMA50 (auf 4h-Basis)
    # Alternativ kann auch über resample('4H') gegangen werden, aber für Backtests reicht das meist.
    df['ema4h'] = df['close'].rolling(window=4, min_periods=1).mean().ewm(span=50, min_periods=1).mean()
    return df

def load_data(parquet_path: str) -> pd.DataFrame:
    """
    Lädt Kursdaten und berechnet alle relevanten Indikatoren.
    Neu: VWAP, MFI, Heikin Ashi, EMA4h.
    """
    df = pd.read_parquet(parquet_path)
    if 'timestamp' not in df:
        df['timestamp'] = pd.date_range("2020-01-01", periods=len(df), freq="h")
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    df['rsi'] = df['close'].rolling(14).apply(
        lambda x: 100 - 100/(1 + (x.diff().clip(lower=0).mean() / abs(x.diff().clip(upper=0).mean()+1e-9))) if len(x) == 14 else np.nan)
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    df['adx'] = abs(df['high'] - df['low']).rolling(14).mean()
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    width = df['bb_upper'] - df['bb_lower']
    df['bb_width'] = width
    df['ema20'] = df['close'].ewm(span=20).mean()
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].replace(0, np.nan).cumsum()
    # MFI
    df['mfi'] = money_flow_index(df)
    # Heikin Ashi
    ha = heikin_ashi(df)
    df['ha_open'] = ha['open']
    df['ha_close'] = ha['close']
    df['ha_high'] = ha['high']
    df['ha_low'] = ha['low']
    # Multi-TF EMA (4h auf 1h)
    df = add_ema4h(df)
    return df

if __name__ == "__main__":
    print(f"[{SKRIPT_ID}] Test: Daten laden und Indikatoren berechnen")
    df = load_data("BTCUSDT_1h_1year_ccxt.parquet")
    print(df[['close','vwap','mfi','ha_open','ha_close','ema4h']].head(10))
