# crypto_trading/data/fetch_all.py
import os
import yfinance as yf

SYMBOLS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]
INTERVALS = {
    "1d": "5y",
    "4h": "2y",
    "1h": "1y",
    "15m": "60d"
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")

def fetch_symbol(symbol: str, interval: str, period: str):
    print(f"Lade {symbol} @ {interval} ({period})")
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        print(f"⚠ Keine Daten für {symbol} @ {interval}")
        return
    path = os.path.join(DATA_DIR, symbol.replace("-", "_"))
    os.makedirs(path, exist_ok=True)
    df.to_parquet(os.path.join(path, f"{interval}.parquet"))
    print(f"✓ Gespeichert: {symbol} {interval}")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for symbol in SYMBOLS:
        for interval, period in INTERVALS.items():
            fetch_symbol(symbol, interval, period)

if __name__ == "__main__":
    main()
