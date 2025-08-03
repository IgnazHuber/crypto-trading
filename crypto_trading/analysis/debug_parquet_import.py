import os
import pandas as pd

DATA_PATH = r"C:\Projekte\crypto_trading\crypto_trading\data\raw"
FILENAME = "BTCUSDT_1h_1year_ccxt.parquet"
FILEPATH = os.path.join(DATA_PATH, FILENAME)



print("Starte Parquet-Import-Test …")
if not os.path.exists(FILEPATH):
    print(f"Datei nicht gefunden: {FILEPATH}")
else:
    df = pd.read_parquet(FILEPATH)
    print(f"Datei geladen: {FILEPATH}")
    print(f"Shape: {df.shape}")
    print(f"Spalten: {df.columns.tolist()}")
    print(df.head(3))
