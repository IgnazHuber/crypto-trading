# crypto_trading/data/fetch_ccxt_binance_data.py

import ccxt
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

DATA_DIR = Path(__file__).parent / "raw"

# Liste der wichtigsten 50 Coins (USDT-Paare) – Anpassen je nach Wunsch!
TOP50_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "DOGE/USDT", "XRP/USDT", "AVAX/USDT", "TRX/USDT", "LINK/USDT",
    "MATIC/USDT", "DOT/USDT", "WBTC/USDT", "BCH/USDT", "TON/USDT", "LTC/USDT", "UNI/USDT", "SHIB/USDT", "ICP/USDT", "ETC/USDT",
    "LEO/USDT", "FIL/USDT", "OKB/USDT", "DAI/USDT", "ATOM/USDT", "RNDR/USDT", "APT/USDT", "XLM/USDT", "ARB/USDT", "HBAR/USDT",
    "CRO/USDT", "VET/USDT", "MKR/USDT", "INJ/USDT", "GRT/USDT", "AAVE/USDT", "SAND/USDT", "EGLD/USDT", "STX/USDT", "NEAR/USDT",
    "QNT/USDT", "FLOW/USDT", "IMX/USDT", "AXS/USDT", "RPL/USDT", "SNX/USDT", "KAVA/USDT", "CRV/USDT", "FTM/USDT", "PEPE/USDT"
]

# Interval-Mapping: '10m' → '15m'
INTERVALS = {
    "1d":  {"interval": "1d", "lookbacks": ["1 year", "3 months", "1 month"]},
    "1h":  {"interval": "1h", "lookbacks": ["1 year", "3 months", "1 month", "1 week"]},
    "10m": {"interval": "15m", "lookbacks": ["1 week", "1 day"]},  # kein 10m bei ccxt, daher 15m
    "1m":  {"interval": "1m", "lookbacks": ["1 day"]},
}

def get_since_date(lookback_str):
    now = datetime.utcnow()
    if "year" in lookback_str:
        return now - timedelta(days=365)
    if "month" in lookback_str:
        return now - timedelta(days=30 * int(lookback_str.split()[0]))
    if "week" in lookback_str:
        return now - timedelta(weeks=int(lookback_str.split()[0]))
    if "day" in lookback_str:
        return now - timedelta(days=int(lookback_str.split()[0]))
    raise ValueError("Unknown lookback_str")

def fetch_and_save_ccxt(symbol, interval_key, lookback_str, interval_mapped):
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    since = int(get_since_date(lookback_str).timestamp() * 1000)
    filename = DATA_DIR / f"{symbol.replace('/', '')}_{interval_key}_{lookback_str.replace(' ', '')}_ccxt.parquet"
    if filename.exists():
        print(f"Skip {filename} (bereits vorhanden)")
        return
    print(f"Fetching {symbol} {interval_mapped} {lookback_str} via ccxt ...")
    all_ohlcv = []
    limit = 1000
    try:
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval_mapped, since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv += ohlcv
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < limit:
                break
            time.sleep(exchange.rateLimit / 1000)
    except Exception as e:
        print(f"Error fetching {symbol} {interval_mapped}: {e}")
        return

    if not all_ohlcv:
        print(f"Keine Daten für {filename}")
        return

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    filename.parent.mkdir(exist_ok=True)
    df.to_parquet(filename)
    print(f"Gespeichert: {filename}")

if __name__ == "__main__":
    for symbol in TOP50_SYMBOLS:
        for interval_key, params in INTERVALS.items():
            interval_mapped = params["interval"]
            for lookback_str in params["lookbacks"]:
                fetch_and_save_ccxt(symbol, interval_key, lookback_str, interval_mapped)
