import ta
import pandas as pd

def calculate_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sma20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['ema12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema26'] = ta.trend.ema_indicator(df['close'], window=26)
    df['trend'] = 'neutral'
    df.loc[df['sma20'] > df['sma50'], 'trend'] = 'bullish'
    df.loc[df['sma20'] < df['sma50'], 'trend'] = 'bearish'
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
    df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close'] * 100
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    df['regime'] = 'range'
    df.loc[df['adx'] > 25, 'regime'] = 'trending'
    df.loc[df['bb_width'] < df['bb_width'].rolling(20).mean().shift(), 'regime'] = 'low_vol'
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['weekday'] = df.index.weekday
    else:
        df['hour'] = 0
        df['weekday'] = 0
    return df
