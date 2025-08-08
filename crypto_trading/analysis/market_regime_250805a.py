# market_regime_250805a.py
SKRIPT_ID = "market_regime_250805a"
"""
Marktumfeld-/Regime-Erkennung für das Krypto-Framework.
Neu: Squeeze, VWAP, MFI, Heikin Ashi, Trend-Kontexte für Strategien.
"""

import pandas as pd

def classify_market_regime(df: pd.DataFrame) -> pd.DataFrame:
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
    # Neu: Squeeze
    bb_width = df['bb_width']
    regimes['squeeze'] = bb_width < bb_width.rolling(100).quantile(0.10)
    regimes['squeeze_breakout_long'] = regimes['squeeze'] & (df['close'] > df['bb_upper'])
    regimes['squeeze_breakout_short'] = regimes['squeeze'] & (df['close'] < df['bb_lower'])
    # VWAP
    regimes['vwap_long'] = df['close'] > df['vwap']
    regimes['vwap_short'] = df['close'] < df['vwap']
    # MFI
    regimes['mfi_oversold'] = df['mfi'] < 20
    regimes['mfi_overbought'] = df['mfi'] > 80
    # Heikin Ashi
    regimes['ha_trend_long'] = df['ha_close'] > df['ha_open']
    regimes['ha_trend_short'] = df['ha_close'] < df['ha_open']
    return regimes

if __name__ == "__main__":
    print(f"[{SKRIPT_ID}] Test: Market Regime Detection erweitert")
