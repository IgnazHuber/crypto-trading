# tests/test_all.py
import os

from crypto_trading.data.fetch_yfinance import fetch_yfinance_data
from crypto_trading.data.fetch_binance import fetch_binance_data
from crypto_trading.strategy.dummy_strategy import DummyStrategy
from crypto_trading.backtesting.engine import BacktestEngine

def test_fetch_yfinance():
    df = fetch_yfinance_data()
    assert not df.empty

def test_fetch_binance():
    df = fetch_binance_data()
    assert not df.empty

def test_strategy_and_backtest():
    strategy = DummyStrategy()
    trades = strategy.run()
    engine = BacktestEngine(trades)
    summary = engine.summary()
    assert "TotalPnL" in summary
    assert "WinRate" in summary
