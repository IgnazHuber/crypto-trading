import numpy as np
import pandas as pd
import talib

def trend_signals(df: pd.DataFrame, long_only: bool = True):
    # --- Debug-Ausgaben ---
    print("\n[trend_signals] Eingehende Spalten:", df.columns.tolist())
    print("[trend_signals] Zeilen:", len(df))

    # --- Pflichtspalten prüfen ---
    required_cols = ["Close", "High", "Low", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Fehlende Spalte: {col}")

    # --- Werte vorbereiten ---
    close = df["Close"].ffill().bfill().to_numpy(dtype="float64").ravel()
    high = df["High"].ffill().bfill().to_numpy(dtype="float64").ravel()
    low = df["Low"].ffill().bfill().to_numpy(dtype="float64").ravel()
    volume = df["Volume"].fillna(0).to_numpy(dtype="float64").ravel()

    print("[trend_signals] close.shape:", close.shape, "dtype:", close.dtype)

    # --- Abbruch wenn keine Daten ---
    if close.size == 0:
        print("[trend_signals] WARNUNG: keine Kursdaten, gebe leere Signale zurück")
        empty = pd.Series(False, index=df.index)
        return empty, empty, empty, empty

    # --- MACD berechnen ---
    macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    macd_signal_line = macd > macd_signal

    # --- ADX berechnen ---
    adx = talib.ADX(high, low, close, timeperiod=14)
    adx_signal = adx > 20

    # --- Volumenfilter ---
    vol_threshold = np.nanmedian(volume)
    vol_signal = volume > vol_threshold

    # --- Long-Signale ---
    long_entries = macd_signal_line & adx_signal & vol_signal
    long_exits = ~macd_signal_line

    if long_only:
        # Nur Long-Handel
        return (
            pd.Series(long_entries, index=df.index),
            pd.Series(long_exits, index=df.index),
            pd.Series(False, index=df.index),
            pd.Series(False, index=df.index),
        )
    else:
        # Short-Signale (symmetrisch, stark vereinfacht)
        short_entries = (~macd_signal_line) & adx_signal & vol_signal
        short_exits = macd_signal_line

        return (
            pd.Series(long_entries, index=df.index),
            pd.Series(long_exits, index=df.index),
            pd.Series(short_entries, index=df.index),
            pd.Series(short_exits, index=df.index),
        )
