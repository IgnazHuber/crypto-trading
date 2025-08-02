# data/fetch_yfinance.py
import yfinance as yf
import pandas as pd

def fetch_yfinance_data(symbol="BTC-USD", period="1y", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df = df.reset_index()
    df.rename(columns={"Date": "date"}, inplace=True)
    return df
