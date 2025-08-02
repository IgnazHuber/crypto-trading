# crypto_trading/data/fetch_binance_data_max.py

import os
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
from pathlib import Path
import time

DATA_DIR = Path(__file__).parent / "raw"

# Deine Top-50 Coins als USDT-Paar
TOP50_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT", "AVAXUSDT", "TRXUSDT", "LINKUSDT",
    "MATICUSDT", "DOTUSDT", "WBTCUSDT", "BCHUSDT", "TONUSDT", "LTCUSDT", "UNIUSDT", "SHIBUSDT", "ICPUSDT", "ETCUSDT",
    "LEOUSDT", "FILUSDT", "OKBUSDT", "DAIUSDT", "ATOMUSDT", "RNDRUSDT", "APTUSDT", "XLMUSDT", "ARBUSDT", "HBARUSDT",
    "CROUSDT", "VETUSDT", "MKRUSDT", "INJUSDT", "GRTUSDT", "AAVEUSDT", "SANDUSDT", "EGLDUSDT", "STXUSDT", "NEARUSDT",
    "QNTUSDT", "FLOWUSDT", "IMXUSDT", "AXSUSDT", "RPLUSDT", "SNXUSDT", "KAVAUSDT", "CRVUSDT", "FTMUSDT", "PEPEUSDT"
]

BINANCE_API_KEY = "3lr8Crn9Ld6dC2qwsGUiBjfZ4pTYGQif4LcSVC21ptlfg9IjLOWkAMprlnPWWwwp"
BINANCE_API_SECRET = ""   # Leer lassen falls nicht nötig, bei nur Read reicht meistens der Key

client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# Spezialfälle: maximal mögliche Historie in Minutenauflösung (i.d.R. 1 Jahr), Tagesdaten bis zu 5 Jahre
INTERVALS = [
    {"name": "max_1m_1year", "interval": Client.KLINE_INTERVAL_1MINUTE, "lookback_days": 365},
    {"name": "max_1d_5years", "interval": Client.KLINE_INTERVAL_1DAY, "lookback_days": 365 * 5},
]

def fetch_and_save(symbol, interval, lookback_days, name):
    filename = DATA_DIR / f"{symbol}_{name}.parquet"
    if filename.exists():
        print(f"Skip {filename} (bereits vorhanden)")
        return
    print(f"Fetching {symbol}, {interval}, {lookback_days}d ...")
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=lookback_days)
    klines = []
    delta = timedelta(days=30 if interval == Client.KLINE_INTERVAL_1MINUTE else 120)
    temp_start = start_dt

    while temp_start < end_dt:
        temp_end = min(temp_start + delta, end_dt)
        try:
            part = client.get_historical_klines(
                symbol, interval,
                temp_start.strftime("%d %b, %Y %H:%M:%S"),
                temp_end.strftime("%d %b, %Y %H:%M:%S"),
                limit=1000 if interval == Client.KLINE_INTERVAL_1MINUTE else 1500
            )
            if part:
                klines.extend(part)
        except Exception as ex:
            print(f"Error {symbol} {interval} {temp_start} - {temp_end}: {ex}")
        temp_start = temp_end
        time.sleep(1.2)
    if not klines:
        print(f"Keine Daten für {symbol} ({name})")
        return
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    filename.parent.mkdir(exist_ok=True)
    df.to_parquet(filename)
    print(f"Gespeichert: {filename}")

if __name__ == "__main__":
    for symbol in TOP50_SYMBOLS:
        for intv in INTERVALS:
            fetch_and_save(
                symbol=symbol,
                interval=intv["interval"],
                lookback_days=intv["lookback_days"],
                name=intv["name"]
            )
