# core/regimes.py

import pandas as pd

def classify_market_regime(df):
    regimes = pd.DataFrame(index=df.index)
    regimes['uptrend'] = (df['ema50'] > df['ema200']) & (df['macd'] > 0)
    regimes['downtrend'] = (df['ema50'] < df['ema200']) & (df['macd'] < 0)
    regimes['sideways'] = (df['adx'] < 20)
    regimes['high_volatility'] = (df['atr'] > df['atr'].rolling(100).median())
    regimes['breakout'] = (df['bb_width'] > df['bb_width'].rolling(100).quantile(0.75))
    regimes['overbought'] = (df['rsi'] > 70)
    regimes['oversold'] = (df['rsi'] < 30)
    regimes['bb_upper'] = df['close'] > df['bb_upper']
    regimes['trend'] = (df['ema20'] > df['ema50']) & (df['ema50'] > df['ema200'])
    regimes['overbought_or_oversold'] = (df['rsi'] > 80) | (df['rsi'] < 20)
    return regimes
