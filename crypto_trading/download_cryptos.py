# crypto_trading/download_cryptos.py

import os
import pandas as pd
import ccxt
from crypto_trading.config import ASSETS, DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)
exchange = ccxt.binance()

def fetch_binance_ohlcv(symbol, since=None, limit=1000):
    binance_symbol = symbol.replace('-USD', '/USDT')
    ohlcv = exchange.fetch_ohlcv(binance_symbol, timeframe='1d', since=since, limit=limit)
    # Binance returns: [timestamp, open, high, low, close, volume]
    df = pd.DataFrame(ohlcv, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], unit="ms")
    return df

for asset in ASSETS:
    try:
        df = fetch_binance_ohlcv(asset)
        if df.empty:
            print(f"Warnung: Keine Daten für {asset}!")
            continue
        save_path = os.path.join(DATA_DIR, f"{asset}.csv")
        df.to_csv(save_path, index=False)
        print(f"OK: {asset} -> {save_path} ({len(df)} Zeilen)")
    except Exception as e:
        print(f"Fehler bei {asset}: {e}")
