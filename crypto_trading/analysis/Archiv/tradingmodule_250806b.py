# analysis/tradingmodule_250806b.py

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

SKRIPT_ID = "tradingmodule_250806b"
RESULTS_DIR = "results"
LAST_DIR_FILE = os.path.join(RESULTS_DIR, "last_dir.txt")
os.makedirs(RESULTS_DIR, exist_ok=True)

PARAMS_PATH = os.path.join(RESULTS_DIR, "best_params.json")
with open(PARAMS_PATH, "r") as f:
    global_params = json.load(f)

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


def run_backtest_for_file(parquet_path, trade_every=1):
    asset = os.path.splitext(os.path.basename(parquet_path))[0].split("_")[0]
    freq = os.path.splitext(os.path.basename(parquet_path))[0].split("_")[1]
    logger.info(f"Verarbeite: {asset} ({freq})")
    df = load_data(parquet_path)
    regimes = prepare_regimes(df)
    add_columns_from_result(df, regimes)
    signals = generate_signals(df, regimes, global_params)
    df['meta_long'] = signals['meta_long']
    trades = run_meta_strategy_with_indicators(df, 'meta_long', global_params,
                                               trade_every=trade_every, asset=asset,
                                               strategy_name=f"Meta_Score_{global_params['min_signals']}")
    return asset, trades, df


def build_portfolio_equity(equity_dict):
    """Summiert alle Equities zeitlich auf gemeinsame Zeitachse"""
    merged = pd.DataFrame()
    for file_label, df in equity_dict.items():
        df2 = df.copy()
        df2 = df2.rename(columns={"Equity": file_label})
        merged = pd.merge(merged, df2[["Time", file_label]],
                          how="outer", on="Time") if not merged.empty else df2[["Time", file_label]]
    merged = merged.sort_values("Time").fillna(method="ffill").fillna(method="bfill")
    merged["PortfolioEquity"] = merged.drop(columns="Time").sum(axis=1)
    return merged[["Time", "PortfolioEquity"]]


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
        asset, trades, df = run_backtest_for_file(parquet_path, trade_every=1)
        pnl_abs_col = get_colname(trades, "pnl_abs")
        if pnl_abs_col:
            equity = 10000 + trades[pnl_abs_col].cumsum()
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

    # === Portfolio-Kennzahlen ===
    equity_curves = {lbl: df.set_index("Time")["Equity"] for lbl, df in equity_dict.items()}
    metrics_df = calculate_portfolio_metrics(equity_curves)
    summary_df = pd.merge(summary_df, metrics_df, left_on="Asset", right_on="Asset", how="left")

    all_trades_df = pd.concat([t for _, t in all_trades_list], ignore_index=True) if all_trades_list else pd.DataFrame()

    # Exporte
    export_to_excel(summary_df, all_trades_df, equity_dict)
    export_to_markdown(summary_df, all_trades_dict)
    export_to_csv_and_html(summary_df, all_trades_df)
    logger.info("Backtest abgeschlossen.")


if __name__ == "__main__":
    main()
