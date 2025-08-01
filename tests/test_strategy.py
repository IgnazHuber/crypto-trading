# tests/test_strategy.py
import pandas as pd
from crypto_trading.indicators.ta_standard import add_indicators
from crypto_trading.strategy.macd_rsi_bollinger import MACD_RSI_Bollinger_Strategy

def test_strategy_runs_without_pandas_ta():
    # Dummy-Daten
    df = pd.DataFrame({
        "Close": [100, 101, 102, 103, 104, 105, 106, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93],
        "Open": [100]*22,
        "High": [102]*22,
        "Low": [98]*22,
        "Volume": [1000]*22
    }, index=pd.date_range("2024-01-01", periods=22))

    df = add_indicators(df).dropna()
    strategy = MACD_RSI_Bollinger_Strategy()
    trades = strategy.run(df)

    # Sicherstellen, dass ein DataFrame zurückgegeben wird
    assert isinstance(trades, pd.DataFrame)

    # Wenn Trades vorhanden sind, Spalten prüfen
    if not trades.empty:
        assert set(["entry_price", "exit_price"]).issubset(trades.columns)
