# crypto_trading/indicators/ta_standard.py
import pandas as pd
import numpy as np

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Berechnet MACD, Signallinie und Histogramm."""
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, min_periods=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def compute_bollinger(series: pd.Series, length: int = 20, num_std: float = 2.0):
    """Berechnet Bollinger Bänder."""
    sma = series.rolling(window=length, min_periods=length).mean()
    std = series.rolling(window=length, min_periods=length).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt RSI, MACD (Linie + Signal + Histogramm) und Bollinger Bänder hinzu.
    Erwartet Spalte 'Close'.
    """
    df = df.copy()

    # RSI
    df["RSI"] = compute_rsi(df["Close"])

    # MACD
    macd, macds, macdh = compute_macd(df["Close"])
    df["MACD_12_26_9"] = macd
    df["MACDs_12_26_9"] = macds
    df["MACDh_12_26_9"] = macdh

    # Bollinger Bänder
    upper, middle, lower = compute_bollinger(df["Close"])
    df["BBU_20_2.0"] = upper
    df["BBM_20_2.0"] = middle
    df["BBL_20_2.0"] = lower

    return df
