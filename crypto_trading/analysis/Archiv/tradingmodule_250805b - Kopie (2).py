# analysis/tradingmodule_250805b.py

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
from .mod_plots import plot_dual_chart_with_markers

SKRIPT_ID = "tradingmodule_250805b"
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
    """Tabnamen für Excel auf max 31 Zeichen und nur gültige Zeichen bringen"""
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


def calc_kpis(trades):
    start_capital = 10000
    pnl_abs_col = get_colname(trades, "pnl_abs")
    pnl_pct_col = get_colname(trades, "pnl_pct")

    if trades.empty:
        return {"Trades": 0, "Total_PnL": 0, "Average_PnL": 0, "Winrate(%)": 0,
                "Start_Capital": start_capital, "End_Capital": start_capital,
                "Max_Drawdown": 0, "Sharpe": None, "CAGR(%)": None, "Volatility(%)": None}

    total_pnl = trades[pnl_abs_col].sum() if pnl_abs_col else 0
    avg_pnl = trades[pnl_abs_col].mean() if pnl_abs_col else 0
    winrate = (trades[pnl_abs_col] > 0).mean() * 100 if pnl_abs_col else 0
    end_capital = start_capital + total_pnl
    max_drawdown = trades["drawdown"].min() if "drawdown" in trades.columns else 0

    sharpe = volatility = None
    if pnl_pct_col and len(trades) > 1:
        returns = trades[pnl_pct_col]
        sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)
        volatility = np.sqrt(252) * returns.std() * 100

    cagr = None
    if "exit_time" in trades.columns and "entry_time" in trades.columns:
        t0 = pd.to_datetime(trades["entry_time"].min())
        t1 = pd.to_datetime(trades["exit_time"].max())
        years = max((t1 - t0).days / 365.25, 0.0001)
        cagr = ((end_capital / start_capital) ** (1 / years) - 1) * 100

    return {"Trades": len(trades), "Total_PnL": total_pnl, "Average_PnL": avg_pnl,
            "Winrate(%)": winrate, "Start_Capital": start_capital, "End_Capital": end_capital,
            "Max_Drawdown": max_drawdown, "Sharpe": sharpe, "CAGR(%)": cagr, "Volatility(%)": volatility}


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


def export_png_equity(file_label, equity_df):
    fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    ax[0].plot(equity_df["Time"], equity_df["Equity"], label="Equity")
    ax[0].set_title(f"{file_label} Equity")
    ax[0].grid(True)
    ax[1].plot(equity_df["Time"], equity_df["Drawdown"], color="red", label="Drawdown")
    ax[1].set_title(f"{file_label} Drawdown")
    ax[1].grid(True)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"equity_{file_label}.png")
    plt.savefig(path)
    plt.close(fig)
    logger.info(f"Equity/Drawdown PNG gespeichert: {path}")


def export_summary(summary_df, all_trades_df, equity_dict):
    out_path = os.path.join(RESULTS_DIR, "trades_summary_all.xlsx")
    summary_df = summary_df.sort_values("Parquet_File").reset_index(drop=True)

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        all_trades_df.to_excel(writer, sheet_name="All_Trades", index=False)

        workbook = writer.book
        summary_ws = writer.sheets["Summary"]

        # Serie für Gewinne (>=10000)
        win_values = [val if val >= 10000 else None for val in summary_df["End_Capital"]]
        # Serie für Verluste (<10000)
        lose_values = [val if val < 10000 else None for val in summary_df["End_Capital"]]

        # Gewinne grün
        chart_endcap = workbook.add_chart({'type': 'column'})
        chart_endcap.add_series({
            'name': "Gewinne",
            'categories': ["Summary", 1, 1, len(summary_df), 1],
            'values':     ["Summary", 1, 7, len(summary_df), 7],
            'fill': {'color': '#00AA00'}
        })
        # Verluste rot (separat)
        chart_endcap.add_series({
            'name': "Verluste",
            'categories': ["Summary", 1, 1, len(summary_df), 1],
            'values':     ["Summary", 1, 7, len(summary_df), 7],
            'fill': {'color': '#CC0000'}
        })
        chart_endcap.set_title({'name': "Endkapital pro Datei"})
        summary_ws.insert_chart("L2", chart_endcap)

        # Equity Tabs
        for file_label, equity_df in equity_dict.items():
            safe_name = safe_tab_name(f"Equity_{file_label}")
            equity_df.to_excel(writer, sheet_name=safe_name, index=False)
            eq_ws = writer.sheets[safe_name]

            rows = len(equity_df)
            if rows < 2:
                continue

            chart_equity = workbook.add_chart({'type': 'line'})
            chart_equity.add_series({
                'name': safe_name,
                'categories': [safe_name, 1, 0, rows, 0],
                'values':     [safe_name, 1, 1, rows, 1]
            })
            chart_equity.set_title({'name': f"Equity {file_label}"})
            eq_ws.insert_chart("E2", chart_equity)

            chart_drawdown = workbook.add_chart({'type': 'line'})
            chart_drawdown.add_series({
                'name': f"{file_label} Drawdown",
                'categories': [safe_name, 1, 0, rows, 0],
                'values':     [safe_name, 1, 2, rows, 2]
            })
            chart_drawdown.set_title({'name': f"Drawdown {file_label}"})
            eq_ws.insert_chart("E20", chart_drawdown)

    logger.info(f"Excel mit Diagrammen gespeichert: {out_path}")


def export_markdown(summary_df, all_trades_dict):
    lines = ["# Trades Summary mit KPIs\n", summary_df.to_markdown(index=False), "\n"]
    for file_label, trades in all_trades_dict.items():
        lines.append(f"## {file_label} – erste 5 Trades\n")
        lines.append(trades.head(5).to_markdown(index=False))
        lines.append("\n")
    md_txt = "\n".join(lines)
    for ext in ["md", "txt"]:
        path = os.path.join(RESULTS_DIR, f"trades_summary_all.{ext}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_txt)
    logger.info("Markdown/TXT-Zusammenfassung gespeichert")


def main():
    batch_mode = "--all" in sys.argv
    parquet_files = select_parquet_files() if not batch_mode else [
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
        timestamps = trades["exit_time"] if "exit_time" in trades.columns else (df["timestamp"] if "timestamp" in df.columns else range(len(equity)))
        equity_df = pd.DataFrame({"Time": timestamps, "Equity": equity, "Drawdown": drawdown})
        equity_dict[file_label] = equity_df
        export_png_equity(file_label, equity_df)

        all_trades_dict[file_label] = trades
        kpis = calc_kpis(trades)
        summary_records.append({"Asset": asset, "Parquet_File": file_label, **kpis})
        if not trades.empty:
            trades["Asset"] = asset
            all_trades_list.append((file_label, trades))

    summary_df = pd.DataFrame(summary_records)
    all_trades_df = pd.concat([t for _, t in all_trades_list], ignore_index=True) if all_trades_list else pd.DataFrame()

    export_summary(summary_df, all_trades_df, equity_dict)
    export_markdown(summary_df, all_trades_dict)
    logger.info("Backtest abgeschlossen.")


if __name__ == "__main__":
    main()
