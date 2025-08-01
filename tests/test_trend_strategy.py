# tests/test_trend_strategy.py
import pandas as pd
from crypto_trading.strategies.trend_macd_adx_volume import trend_signals

def test_trend_signals_shape():
    df = pd.DataFrame({
        "Open": [1,2,3,4,5],
        "High": [2,3,4,5,6],
        "Low": [1,2,2,3,4],
        "Close": [1.5,2.5,3.5,4.5,5.5],
        "Volume": [100,200,150,300,400]
    })
    entries, exits = trend_signals(df)
    assert len(entries) == len(df)
    assert len(exits) == len(df)
