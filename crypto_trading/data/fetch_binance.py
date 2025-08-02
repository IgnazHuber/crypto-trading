# data/fetch_binance.py
import pandas as pd
import numpy as np

def fetch_binance_data(symbol="BTCUSDT"):
    # Dummy: Erzeuge 30 Tage Zufallsdaten
    dates = pd.date_range(start="2024-01-01", periods=30)
    prices = np.linspace(40000, 45000, 30) + np.random.randn(30)*500
    return pd.DataFrame({"date": dates, "open": prices, "high": prices+100, 
                         "low": prices-100, "close": prices, "volume": np.random.randint(100,1000,30)})
