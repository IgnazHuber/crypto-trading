# analysis/tradingmodule_250806f.py

import os
import sys
import json
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import logging
import matplotlib.pyplot as plt
import re

from .mod_data import load_data
from .mod_utils import add_columns_from_result
from .mod_strategy import prepare_regimes, generate_signals
from .mod_trades import run_meta_strategy_with_indicators
from .portfolio_metrics import calculate_portfolio_metrics
from .mod_excel import export_to_excel
from .mod_markdown import export_to_markdown
from .mod_csv_html import export_to_csv_and_html
from .mod_png import export_png_equity

SKRIPT_ID = "tradingmodule_250806f"
RESULTS_DIR = "results"
LAST_DIR_FILE = os.path.join(RESULTS_DIR, "last_dir.txt")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Defaultwerte für Parameter ---
DEFAULT_PARAMS = {
    "min_signals": 1,
    "ema_length": 20,
    "sma_length": 50,
    "rsi_length": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "mfi_length": 14,
    "atr_length": 14,
    "bb_length": 20,
    "bb_std": 2.0
}

# --- Parameter laden (best_params.json) ---
PARAMS_PATH = os.path.join(RESULTS_DIR, "best_params.json")
global_params = {}
if os.path.exists(PARAMS_PATH):
    try:
        with open(PARAMS_PATH, "r") as f:
            global_params = json.load(f)
        print(f"[INFO] Optimierte Parameter aus {PARAMS_PATH} geladen.")
    except Exception as e:
        print(f"[WARN] Fehler beim Laden von best_params.json ({e}), verwende nur Defaultparameter.")
else:
    print("[WARN] Keine best_params.json gefunden, verwende nur Defaultparameter.")

# Fehlende Keys ergänzen und tracken, was Default ist
param_origin = {}
for key, default_val in DEFAULT_PARAMS.items():
    if key not in global_params:
        global_params[key] = default_val
        param_origin[key] = "default"
        print(f"[INFO] Standardwert ergänzt: {key} = {default_val}")
    else:
        param_origin[key] = "optimized"

log_path = os.path.join(RESULTS_DIR, f"log_{SKRIPT_ID}.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_path, mode='w', encoding='utf-8'),
              logging.StreamHandler()]
)
logger = logging.getLogger(SKRIPT_ID)


def safe_tab_name(name):
    return re.sub(r'[^A-Za-z0-9_]', '_', name)[:25]


def select_parquet_files():
    if os.path.exists(LAST_DIR_FILE):
        with open(LAST_DIR_FILE, "r") as f:
            start_dir = f.read().strip()
    else:
        start_dir = r"D:\Projekte\crypto_trading\crypto_trading\data\raw"
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Wähle eine oder mehrere Parquet-Dateien",
        initialdir=start_dir,
        filetypes=[("Parquet Dateien", "*.parquet")],
    )
    if file_paths:
        last_dir = os.path.dirname(file_paths[0])
        with open(LAST_DIR_FILE, "w") as f:
            f.write(last_dir)
    return list(file_paths)


def get_colname(trades, target_name):
    return next((c for c in trades.columns if c.lower() == target_name.lower()), None)


def run_backtest_for_file(parquet_path, trade_every=1, params=None):
    params = params or {}
    asset = os.path.splitext(os.path.basename(parquet_path))[0].split("_")[0]
    freq = os.path.splitext(os.path.basename(parquet_path))[0].split("_")[1]
    logger.info(f"Verarbeite: {asset} ({freq}) mit Parametern: {params}")
    df = load_data(parquet_path)
    regimes = prepare_regimes(df)
    add_columns_from_result(df, regimes)
    signals = generate_signals(df, regimes, params)
    df['meta_long'] = signals['meta_long']
    trades = run_meta_strategy_with_indicators(df, 'meta_long', params,
                                               trade_every=trade_every, asset=asset,
                                               strategy_name=f"Meta_Score")
    return asset, trades, df


def build_portfolio_equity(equity_dict):
    merged = pd.DataFrame()
    for file_label, df in equity_dict.items():
        df2 = df.copy()
        df2 = df2.rename(columns={"Equity": file_label})
        merged = pd.merge(merged, df2[["Time", file_label]],
                          how="outer", on="Time") if not merged.empty else df2[["Time", file_label]]
    merged = merged.sort_values("Time").fillna(method="ffill").fillna(method="bfill")
    merged["PortfolioEquity"] = merged.drop(columns="Time").sum(axis=1)
    return merged[["Time", "PortfolioEquity"]]


def export_params_to_excel(writer, params: dict, origin: dict):
    df_params = pd.DataFrame([
        {"Parameter": k, "Wert": v, "Quelle": "default" if origin[k] == "default" else "optimized"}
        for k, v in params.items()
    ])
    df_params.to_excel(writer, sheet_name="Used_Params", index=False)


def export_params_to_markdown(params: dict, origin: dict, path="results/used_params.md"):
    lines = ["# Verwendete Parameter (aus best_params.json + Defaults)", ""]
    for k, v in params.items():
        src = "(default)" if origin[k] == "default" else "(optimized)"
        lines.append(f"- **{k}**: {v} {src}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Parameter-Markdown gespeichert: {path}")


def main():
    parquet_files = select_parquet_files() if "--all" not in sys.argv else [
        os.path.join(r"D:\Projekte\crypto_trading\crypto_trading\data\raw", f)
        for f in os.listdir(r"D:\Projekte\crypto_trading\crypto_trading\data\raw") if f.endswith(".parquet")
    ]
    if not parquet_files:
        logger.warning("Keine Dateien ausgewählt.")
        return

    all_trades_list, equity_dict, all_trades_dict = [], {}, {}
    summary_records = []

    for parquet_path in parquet_files:
        file_label = os.path.splitext(os.path.basename(parquet_path))[0]
        asset, trades, df = run_backtest_for_file(parquet_path, trade_every=1, params=global_params)
        pnl_abs_col = get_colname(trades, "pnl_abs")
        if pnl_abs_col:
            equity = 10000 + trades[pnl_abs_col].cumsum()
            # Falls Start vor erstem Trade gezeigt werden soll:
            if len(equity) > 0:
                equity.iloc[0] = 10000
        else:
            equity = pd.Series([10000] * len(df))
        drawdown = equity - equity.cummax()
        timestamps = trades["exit_time"] if "exit_time" in trades.columns else (
            df["timestamp"] if "timestamp" in df.columns else range(len(equity)))
        equity_df = pd.DataFrame({"Time": timestamps, "Equity": equity, "Drawdown": drawdown})
        equity_dict[file_label] = equity_df

        export_png_equity(file_label, equity_df)

        all_trades_dict[file_label] = trades
        summary_records.append({"Asset": asset, "Parquet_File": file_label,
                                "End_Capital": equity.iloc[-1], "Trades": len(trades)})

        if not trades.empty:
            trades["Asset"] = asset
            all_trades_list.append((file_label, trades))

    summary_df = pd.DataFrame(summary_records)
    equity_curves = {lbl: df.set_index("Time")["Equity"] for lbl, df in equity_dict.items()}
    metrics_df = calculate_portfolio_metrics(equity_curves)
    summary_df = pd.merge(summary_df, metrics_df, left_on="Asset", right_on="Asset", how="left")

    all_trades_df = pd.concat([t for _, t in all_trades_list], ignore_index=True) if all_trades_list else pd.DataFrame()

    # Exporte mit Parametern
    excel_path = os.path.join(RESULTS_DIR, "trades_summary_all.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        all_trades_df.to_excel(writer, sheet_name="All_Trades", index=False)
        for file_label, equity_df in equity_dict.items():
            equity_df.to_excel(writer, sheet_name=safe_tab_name(file_label)[:31], index=False)
        export_params_to_excel(writer, global_params, param_origin)
    export_params_to_markdown(global_params, param_origin, path=os.path.join(RESULTS_DIR, "used_params.md"))
    export_to_markdown(summary_df, all_trades_dict)
    export_to_csv_and_html(summary_df, all_trades_df)
    logger.info("Backtest abgeschlossen.")


if __name__ == "__main__":
    main()
