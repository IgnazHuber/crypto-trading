# analysis/mod_indicators.py

import pandas as pd

def calc_indicators(df):
    """Berechnet alle benötigten Indikatoren für Regimes, Strategie, Hover."""

    # Grundspalten prüfen
    need = ['close','high','low']
    missing = [x for x in need if x not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Grundspalten für Indikatoren: {missing} in DataFrame: {df.columns.tolist()}")

    # EMA50, EMA200, EMA20
    if 'ema50' not in df.columns:
        df['ema50'] = df['close'].rolling(50, min_periods=1).mean()
    if 'ema200' not in df.columns:
        df['ema200'] = df['close'].rolling(200, min_periods=1).mean()
    if 'ema20' not in df.columns:
        df['ema20'] = df['close'].rolling(20, min_periods=1).mean()
    # MACD
    if 'macd' not in df.columns:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
    # ATR
    if 'atr' not in df.columns:
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14, min_periods=1).mean()
    # Bollinger-Bänder
    if 'bb_width' not in df.columns:
        ma = df['close'].rolling(20, min_periods=1).mean()
        std = df['close'].rolling(20, min_periods=1).std()
        df['bb_upper'] = ma + 2 * std
        df['bb_lower'] = ma - 2 * std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / ma
    # RSI
    if 'rsi' not in df.columns:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
    # MFI
    if 'mfi' not in df.columns and {'high','low','close','volume'}.issubset(df.columns):
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        pos_mf = mf.where(tp > tp.shift(), 0).rolling(14, min_periods=1).sum()
        neg_mf = mf.where(tp < tp.shift(), 0).rolling(14, min_periods=1).sum()
        mfr = pos_mf / (neg_mf + 1e-8)
        df['mfi'] = 100 - (100 / (1 + mfr))
    # VWAP
    if 'vwap' not in df.columns and {'close','volume'}.issubset(df.columns):
        df['vwap'] = (df['close'] * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-8)
    # Heikin-Ashi
    if 'ha_close' not in df.columns and {'open','high','low','close'}.issubset(df.columns):
        df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    if 'ha_open' not in df.columns and 'ha_close' in df:
        df['ha_open'] = df['open'].expanding().apply(lambda x: (x.iloc[0] + x.iloc[-1]) / 2)
    # ADX
    if 'adx' not in df.columns:
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        atr = tr.rolling(14, min_periods=1).mean()
        plus_di = 100 * (plus_dm.rolling(14, min_periods=1).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(14, min_periods=1).sum() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['adx'] = dx.rolling(14, min_periods=1).mean()

    print("Spalten nach calc_indicators:", df.columns.tolist())
    return df
