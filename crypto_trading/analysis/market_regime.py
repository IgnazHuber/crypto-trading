# crypto_trading/analysis/market_regime.py

import pandas as pd

def detect_market_regime(df, window=20):
    """
    Klassifiziert das Marktumfeld pro Tag (Trend, Range, Volatil)
    df: DataFrame mit 'ADX', 'ATR' etc.
    Gibt Series mit Regime-Labels zurück.
    """
    atr_rolling = df['ATR'].rolling(window).mean()
    regimes = []
    for idx, row in df.iterrows():
        adx = row.get('ADX', None)
        atr = row.get('ATR', None)
        atr_roll = atr_rolling.loc[idx] if idx in atr_rolling.index else None
        if adx is not None and adx > 25:
            regimes.append('trend')
        elif atr is not None and atr_roll is not None and atr > atr_roll * 1.5:
            regimes.append('volatile')
        else:
            regimes.append('range')
    return pd.Series(regimes, index=df.index, name="Regime")
