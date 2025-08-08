import os
import sys
import json
import tkinter as tk
from tkinter import filedialog
import pandas as pd

from .indicator_library import calc_all_indicators
from .indicator_context import classify_trend, classify_volatility, classify_volume, apply_context_to_indicators
from .mod_trades import run_meta_strategy_with_indicators
from .mod_plots import export_png_equity
from .report_export import (
    export_excel_summary,
    export_html_reports,
    export_markdown_params,
    export_csv_trades
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
PARAMS_PATH = os.path.join(RESULTS_DIR, "best_params.json")


def select_files():
    """Öffnet einen Datei-Dialog zur Auswahl mehrerer Parquet-Dateien."""
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Wähle Parquet-Dateien",
        initialdir=r"D:\Projekte\crypto_trading\crypto_trading\data\raw",
        filetypes=[("Parquet Dateien", "*.parquet")]
    )
    return file_paths


def main():
    # --- Parquet-Dateien auswählen ---
    file_paths = select_files()
    if not file_paths:
        print("[Trading] Keine Dateien ausgewählt – Abbruch.")
        return

    # --- Parameter laden ---
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, "r") as f:
            best_params_map = json.load(f)
        if "min_signals" not in best_params_map:
            best_params_map["min_signals"] = 1
        print(f"[Trading] Parameter geladen aus {PARAMS_PATH}")
    else:
        print("[Trading] WARN: Keine best_params.json gefunden, nutze Standardparameter.")
        best_params_map = {"sideways": {}, "uptrend": {}, "downtrend": {}, "min_signals": 1}

    for file_path in file_paths:
        print(f"[Trading] Verarbeite Datei: {file_path}")
        df = pd.read_parquet(file_path)
        df = calc_all_indicators(df, {})
        df = classify_trend(df)
        df = classify_volatility(df)
        df = classify_volume(df)
        df = apply_context_to_indicators(df)

        # --- Backtest mit adaptiver Meta-Strategie ---
        trades_df, end_capital, max_loss = run_meta_strategy_with_indicators(
            df, "meta_long", best_params_map, trade_every=1,
            start_capital=10000.0,
            max_risk_per_trade=0.10,
            sl_pct=0.03,
            tp_pct=0.20
        )

        asset_name = os.path.splitext(os.path.basename(file_path))[0]
        excel_path = os.path.join(RESULTS_DIR, f"trades_summary_{asset_name}.xlsx")
        html_path = os.path.join(RESULTS_DIR, f"trades_summary_{asset_name}.html")
        markdown_path = os.path.join(RESULTS_DIR, f"used_params_{asset_name}.md")
        csv_path = os.path.join(RESULTS_DIR, f"all_trades_{asset_name}.csv")
        png_path = os.path.join(RESULTS_DIR, f"equity_curve_{asset_name}.png")

        export_excel_summary(trades_df, end_capital, max_loss, excel_path)
        export_html_reports(trades_df, html_path)
        export_markdown_params(best_params_map, markdown_path)
        export_csv_trades(trades_df, csv_path)
        export_png_equity(trades_df, png_path)

        print(f"[Trading] {asset_name}: Endkapital = {end_capital:.2f}, Max. Verlust = {max_loss:.2f}")
        print(f"[Trading] Exportiert: Excel={excel_path}, HTML={html_path}, PNG={png_path}")


if __name__ == "__main__":
    main()
