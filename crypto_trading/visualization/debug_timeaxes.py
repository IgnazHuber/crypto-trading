import os
import pandas as pd

# --- Pfade anpassen falls nötig ---
PARQUET_PATH = r"d:\Projekte\crypto_trading\crypto_trading\data\raw\BTCUSDT_1h_1year_ccxt.parquet"
EXCEL_PATH = r"d:\Projekte\crypto_trading\results\btc_candlestick_backtest_regime.xlsx"

print("\n==> Kursdaten (Candlesticks) <==")
df = pd.read_parquet(PARQUET_PATH)
if 'timestamp' not in df.columns:
    if isinstance(df.index, pd.DatetimeIndex):
        df['timestamp'] = df.index
    else:
        print("Kursdaten enthalten keine Zeitspalte!")
        exit(1)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
print("Min timestamp:", df['timestamp'].min())
print("Max timestamp:", df['timestamp'].max())
print(df[['timestamp', 'open', 'close']].head(3))
print(df[['timestamp', 'open', 'close']].tail(3))

print("\n==> Trades aus Excel <==")
df_all = pd.read_excel(EXCEL_PATH, sheet_name=None)
for strat, trades in df_all.items():
    if strat in ("KPIs", "Alle Trades") or trades.empty:
        continue
    # Spalten vereinheitlichen
    for col in trades.columns:
        if col.lower().replace(" ", "").replace("_", "") == "entrytime":
            trades.rename(columns={col: "Entry Time"}, inplace=True)
        if col.lower().replace(" ", "").replace("_", "") == "exittime":
            trades.rename(columns={col: "Exit Time"}, inplace=True)
    trades['Entry Time'] = pd.to_datetime(trades['Entry Time'], errors='coerce')
    trades['Exit Time'] = pd.to_datetime(trades['Exit Time'], errors='coerce')
    print(f"\n--- Strategie: {strat} ---")
    print("Min Entry Time:", trades['Entry Time'].min())
    print("Max Entry Time:", trades['Entry Time'].max())
    print("Min Exit Time:", trades['Exit Time'].min())
    print("Max Exit Time:", trades['Exit Time'].max())
    print("Beispiel Trades:")
    print(trades[['Entry Time', 'Exit Time']].head(3))
    print(trades[['Entry Time', 'Exit Time']].tail(3))

print("\n[INFO] --> Bitte prüfe die Zeiträume und poste ggf. den Output!")
