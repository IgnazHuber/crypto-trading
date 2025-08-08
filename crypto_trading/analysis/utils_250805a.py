# utils.py
SKRIPT_ID = "utils_250805a"
"""
Hilfsfunktionen, Mapping von Spaltennamen, Sheet-Namen, etc.
"""

import re

def ensure_legacy_trade_columns(df):
    column_map = {
        "entry_time": "Entry Time", "entry_date": "Entry Time", "Entry": "Entry Time",
        "exit_time": "Exit Time", "exit_date": "Exit Time", "Exit": "Exit Time",
        "entry_price": "Entry Price", "EntryPrice": "Entry Price",
        "exit_price": "Exit Price", "ExitPrice": "Exit Price",
        "trade_id": "Trade-ID", "TradeId": "Trade-ID",
        "asset": "Asset", "symbol": "Asset",
        "einsatz": "Einsatz",
        "kapital_nach_trade": "Kapital nach Trade", "KapitalNachTrade": "Kapital nach Trade",
        "pnl_abs": "PnL_abs", "pnl": "PnL_abs", "PnL": "PnL_abs",
        "pnl_pct": "PnL_pct", "PnL_pct": "PnL_pct",
        "strategy": "Strategy", "regime": "Regime",
    }
    rename_map = {k: v for k, v in column_map.items() if k in df.columns}
    return df.rename(columns=rename_map)

def safe_sheet_name(name, maxlen=28):
    name = re.sub(r'[\/\\\?\*\[\]\:]', '_', name)
    return name[:maxlen]

if __name__ == "__main__":
    print(f"[{SKRIPT_ID}] Test: safe_sheet_name('Test/Sheet:Name*')")
    print(safe_sheet_name("Test/Sheet:Name*"))
