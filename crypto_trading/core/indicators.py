# core/indicators.py

import numpy as np

def add_indicators(df):
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    df['rsi'] = df['close'].rolling(14).apply(
        lambda x: 100 - 100/(1 + (x.diff().clip(lower=0).mean() / abs(x.diff().clip(upper=0).mean()+1e-9)))
        if len(x) == 14 else np.nan
    )
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    df['adx'] = abs(df['high'] - df['low']).rolling(14).mean()
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    width = df['bb_upper'] - df['bb_lower']
    df['bb_width'] = width
    df['ema20'] = df['close'].ewm(span=20).mean()
    return df
