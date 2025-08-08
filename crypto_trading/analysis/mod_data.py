# analysis/mod_data.py

import os
import re
import pandas as pd

DATA_RAW_DIR = r"d:\Projekte\crypto_trading\crypto_trading\data\raw"

def scan_rawdata_files(directory=DATA_RAW_DIR):
    files = [f for f in os.listdir(directory) if f.endswith(".parquet")]
    data_list = []
    for file in files:
        m = re.match(r"([A-Z0-9]+(?:_[A-Z0-9]+)?)[-_](\d+[mhdw])_", file)
        if m:
            asset = m.group(1).replace("_", "/")
            freq = m.group(2)
        else:
            asset, freq = file.split("_")[0], "?"
        data_list.append({"file": file, "asset": asset, "freq": freq})
    return data_list

def choose_rawdata():
    data_list = scan_rawdata_files()
    print("\n--- Verfügbare Rohdaten ---")
    for i, d in enumerate(data_list):
        print(f"{i}: {d['asset']} ({d['freq']}) [{d['file']}]")
    idx = int(input("Nummer der Datei für den Backtest wählen: "))
    dsel = data_list[idx]
    return dsel['file'], dsel['asset'], dsel['freq']

def choose_trading_frequency(base_freq):
    m = re.match(r"(\d+)([mhdw])", base_freq)
    options = [1, 2, 4, 8]
    if m:
        unit = m.group(2)
        num = int(m.group(1))
        print("\n--- Tradingfrequenz (relativ zur Datenfrequenz) ---")
        for i, mult in enumerate(options):
            print(f"{i}: Jede {mult}te Candle ({num*mult}{unit})")
        idx = int(input("Nummer der Tradingfrequenz wählen: "))
        return options[idx]
    else:
        print("Konnte Frequenz nicht erkennen, nehme jede Candle.")
        return 1

def load_data(parquet_path):
    df = pd.read_parquet(parquet_path)
    print("Spalten im geladenen DataFrame:", df.columns.tolist())
    # Robust: timestamp-Spalte erzwingen
    if 'timestamp' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={'index': 'timestamp'})
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            # Fallback: fortlaufender Index als Dummy
            df['timestamp'] = pd.date_range(start="2024-01-01", periods=len(df), freq="H")
        print("[mod_data] timestamp-Spalte automatisch ergänzt:", df['timestamp'].head())
    return df
