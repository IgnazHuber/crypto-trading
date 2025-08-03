# crypto_trading/analysis/trades.py

import pandas as pd
import numpy as np
import os

MODULE_ROOT = os.path.abspath(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.normpath(os.path.join(MODULE_ROOT, "..", "data", "raw"))

def asset_symbol_parquet(asset, to_usdt=True):
    asset = asset.upper().replace("-", "").replace("_", "")
    if to_usdt and asset.endswith("USD") and not asset.endswith("USDT"):
        asset = asset + "T"
    return asset

def parquet_path(asset, timeframe, years, exchange):
    asset_fixed = asset_symbol_parquet(asset)
    return os.path.join(RAW_DATA_DIR, f"{asset_fixed}_{timeframe}_{years}year_{exchange}.parquet")

def load_asset_data(asset, timeframe="1h", years=1, exchange="ccxt"):
    path = parquet_path(asset, timeframe, years, exchange)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Asset-Daten fehlen: {path}")
    df = pd.read_parquet(path)
    date_col = [c for c in df.columns if "date" in c.lower()]
    if date_col:
        df[date_col[0]] = pd.to_datetime(df[date_col[0]])
        df = df.set_index(date_col[0])
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Keine Datumsspalte erkannt in {path}")
    # Typen fixen
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col.capitalize() in df.columns:
            df[col] = pd.to_numeric(df[col.capitalize()], errors="coerce")
    return df

def compute_indicators(df):
    df = df.copy()
    if "Close" in df.columns: df["close"] = df["Close"]
    if "High" in df.columns: df["high"] = df["High"]
    if "Low" in df.columns: df["low"] = df["Low"]
    if "Volume" in df.columns: df["volume"] = df["Volume"]

    df["EMA_FAST"] = df["close"].ewm(span=12, min_periods=12).mean()
    df["EMA_SLOW"] = df["close"].ewm(span=26, min_periods=26).mean()
    df["MACD"] = df["EMA_FAST"] - df["EMA_SLOW"]
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window=14, min_periods=14).mean()
    avg_loss = down.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    sma20 = df["close"].rolling(window=20, min_periods=20).mean()
    std20 = df["close"].rolling(window=20, min_periods=20).std()
    df["BB_MIDDLE"] = sma20
    df["BB_UPPER"] = sma20 + 2 * std20
    df["BB_LOWER"] = sma20 - 2 * std20
    df["ADX"] = df["high"].rolling(14).mean() - df["low"].rolling(14).mean()
    low14 = df["low"].rolling(window=14).min()
    high14 = df["high"].rolling(window=14).max()
    df["STOCH"] = 100 * (df["close"] - low14) / (high14 - low14 + 1e-9)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical_price * df["volume"]
    pos_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    neg_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    pos_mf = pd.Series(pos_flow).rolling(window=14, min_periods=14).sum()
    neg_mf = pd.Series(neg_flow).rolling(window=14, min_periods=14).sum()
    mfi = 100 * pos_mf / (pos_mf + neg_mf + 1e-9)
    df["MFI"] = mfi.values
    df["VOL"] = df["volume"].rolling(window=7, min_periods=7).mean()
    df["OBV"] = (np.sign(df["close"].diff()) * df["volume"].fillna(0)).cumsum()
    df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()
    return df
