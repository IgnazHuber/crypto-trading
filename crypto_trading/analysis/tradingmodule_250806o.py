import os
import json
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from datetime import datetime
from .indicator_library import calc_all_indicators
from .indicator_context import classify_trend, classify_volatility, classify_volume, apply_context_to_indicators
from .mod_trades import run_meta_strategy_with_indicators
from .mod_plots import export_png_equity
from .report_excel import export_excel_summary  # <- bestehend

RESULTS_DIR = "results"
BEST_PARAMS_FILE = os.path.join(RESULTS_DIR, "best_params.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

def select_files():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilenames(
        title="Wähle Parquet-Dateien",
        initialdir=r"D:\Projekte\crypto_trading\crypto_trading\data\raw",
        filetypes=[("Parquet Dateien", "*.parquet")]
    )

def load_best_params():
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE, "r") as f:
            return json.load(f)
    print(f"[WARN] Keine {BEST_PARAMS_FILE} gefunden – Standardparameter werden genutzt.")
    return {}

def determine_context_params(df, best_params_map):
    df = classify_trend(df)
    df = classify_volatility(df)
    df = classify_volume(df)
    df = apply_context_to_indicators(df)
    df["context_params"] = df["market_context"].apply(lambda ctx: best_params_map.get(ctx, {}))
    return df

def main():
    file_paths = select_files()
    if not file_paths:
        print("[Trading] Keine Dateien ausgewählt – Abbruch.")
        return

    best_params_map = load_best_params()

    for file_path in file_paths:
        print(f"[Trading] Verarbeite: {file_path}")
        df = pd.read_parquet(file_path)
        df = calc_all_indicators(df, {})
        df = determine_context_params(df, best_params_map)

        trades_df, end_capital, max_loss = run_meta_strategy_with_indicators(
            df, "meta_long", best_params_map,
            trade_every=1,
            start_capital=10000.0,
            max_risk_per_trade=0.10,
            sl_pct=0.03,
            tp_pct=0.20
        )

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        excel_path = os.path.join(RESULTS_DIR, f"trades_summary_{base_name}_{timestamp}.xlsx")
        export_excel_summary(trades_df, excel_path)
        print(f"[Excel] Exportiert: {excel_path}")

        png_path = os.path.join(RESULTS_DIR, f"equity_{base_name}_{timestamp}.png")
        export_png_equity(trades_df, png_path)
        print(f"[PNG] Exportiert: {png_path}")

if __name__ == "__main__":
    main()
