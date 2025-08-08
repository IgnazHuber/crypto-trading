import os
import pandas as pd

# --- Settings ---
RESULTS_DIR = r"d:\Projekte\crypto_trading\results"
PARQUET_PATH = r"d:\Projekte\crypto_trading\crypto_trading\data\raw\BTCUSDT_1h_1year_ccxt.parquet"
OUTPUT_EXCEL = os.path.join(RESULTS_DIR, "btc_candlestick_backtest_regime.xlsx")
SYNC_TRADES_PATH = os.path.join(RESULTS_DIR, "btc_trades_with_ts.parquet")

# --- Kursdaten laden & Zeitstempel sichern ---
df = pd.read_parquet(PARQUET_PATH)
if 'timestamp' not in df.columns:
    df['timestamp'] = pd.date_range("2020-01-01", periods=len(df), freq="h")
candle_times = pd.Series(df['timestamp']).sort_values().reset_index(drop=True)

# --- Trades laden ---
df_all = pd.read_excel(OUTPUT_EXCEL, sheet_name=None)
trades_synced = []

def standardize_columns(trades):
    mapping = {
        "entrytime": "Entry Time", "exittime": "Exit Time",
        "entryprice": "Entry Price", "exitprice": "Exit Price",
        "tradeid": "trade_id", "einsatz": "Einsatz",
        "pnl_abs": "PnL_abs", "pnlpct": "PnL_pct", "kapitalnachtrade": "Kapital nach Trade"
    }
    rename = {orig: new for orig in trades.columns for k, new in mapping.items() if orig.lower().replace(" ", "").replace("_", "") == k}
    return trades.rename(columns=rename)

for strat, trades in df_all.items():
    if strat in ("KPIs", "Alle Trades") or trades.empty:
        continue
    trades = standardize_columns(trades)
    for col in ["Entry Time", "Exit Time"]:
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col], errors='coerce')
    # Snap Timestamps an Candlestick-Achse
    for col in ["Entry Time", "Exit Time"]:
        if col in trades.columns:
            trades[col + "_snap"] = trades[col].apply(
                lambda x: candle_times.iloc[(candle_times - x).abs().argmin()] if pd.notnull(x) else pd.NaT
            )
    trades["strategy"] = strat
    trades_synced.append(trades)

# --- Zusammenführen und speichern ---
df_merged = pd.concat(trades_synced, ignore_index=True)
print(df_merged[['strategy', 'Entry Time', 'Entry Time_snap', 'Exit Time', 'Exit Time_snap', 'Entry Price', 'Exit Price']].head(10))
missing_entry = df_merged['Entry Time_snap'].isna().sum()
missing_exit = df_merged['Exit Time_snap'].isna().sum()
print(f"Anzahl fehlende Entry Time_snap: {missing_entry}")
print(f"Anzahl fehlende Exit Time_snap: {missing_exit}")

df_merged.to_parquet(SYNC_TRADES_PATH)
print(f"\nSynchronisierte Trades gespeichert: {SYNC_TRADES_PATH}\n")
os._exit(0)
