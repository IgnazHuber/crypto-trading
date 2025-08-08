# strategy_optimizer.py
import os
import sys
import json
import itertools
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .indicator_library import calc_all_indicators
from .indicator_context import classify_trend, classify_volatility, classify_volume, apply_context_to_indicators
from .mod_trades import run_meta_strategy_with_indicators
from .optimizer_visuals import create_all_visuals

RESULTS_DIR = "results"
BEST_PARAMS_PATH = os.path.join(RESULTS_DIR, "best_params.json")


# -------------------------------
# Wichtige Indikatoren (sortiert)
# -------------------------------
INDICATOR_PRIORITY = [
    "ema_length", "sma_length", "rsi_length", "macd_fast", "macd_slow", "macd_signal",
    "mfi_length", "atr_length", "bb_length", "bb_std"
    # Restliche Indikatoren aus Top-50 hier ergänzbar
]


# -------------------------------
# Datei-Auswahl
# -------------------------------
def select_files_dialog():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Wähle Parquet-Dateien für Optimierung",
        initialdir=r"D:\Projekte\crypto_trading\crypto_trading\data\raw",
        filetypes=[("Parquet Dateien", "*.parquet")]
    )
    return list(file_paths)


# -------------------------------
# Dynamische Parameter-Grid Erstellung
# -------------------------------
def generate_param_grid(indicator_count=50, depth=1):
    """
    Erstellt Parameterkombinationen für die wichtigsten Indikatoren.
    depth = 1 → schnell (nur Standardwerte), depth = 2 → vollständiger Grid.
    """
    # Parameterbereiche (Tiefe 2 = alle, Tiefe 1 = ein Wert pro Indikator)
    param_space_full = {
        "ema_length": [20] if depth == 1 else [10, 20, 50],
        "sma_length": [50] if depth == 1 else [20, 50, 100],
        "rsi_length": [14] if depth == 1 else [7, 14, 21],
        "macd_fast": [12] if depth == 1 else [8, 12],
        "macd_slow": [26] if depth == 1 else [17, 26],
        "macd_signal": [9] if depth == 1 else [5, 9],
        "mfi_length": [14] if depth == 1 else [10, 14, 20],
        "atr_length": [14] if depth == 1 else [7, 14],
        "bb_length": [20] if depth == 1 else [14, 20],
        "bb_std": [2.0] if depth == 1 else [1.5, 2.0, 2.5],
    }

    # Wichtigkeit sortieren + Anzahl begrenzen
    selected_keys = INDICATOR_PRIORITY[:max(1, min(indicator_count, len(INDICATOR_PRIORITY)))]
    reduced_space = {k: param_space_full[k] for k in selected_keys}

    keys, values = zip(*reduced_space.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# -------------------------------
# Robustere Trendklassifizierung
# -------------------------------
def safe_classify_trend(df):
    try:
        df = classify_trend(df)
    except ValueError:
        df["trend_class"] = ["sideways"] * len(df)
    return df


# -------------------------------
# Backtest mit Indikatoren & Kontext (globale Funktion für Multiprocessing)
# -------------------------------
def run_backtest_task(args):
    df, params = args
    try:
        df = calc_all_indicators(df.copy(), params)
        df = safe_classify_trend(df)
        df = classify_volatility(df)
        df = classify_volume(df)
        df = apply_context_to_indicators(df)

        trades = run_meta_strategy_with_indicators(df, 'meta_long', params, trade_every=1, asset="OPTIMIZER")
        end_capital = 10000 + trades["pnl_abs"].sum() if not trades.empty else 10000
        max_single_loss = trades["pnl_abs"].min() if not trades.empty else 0
        return {
            "params": params,
            "end_capital": end_capital,
            "max_single_loss": max_single_loss,
            "trades": len(trades)
        }
    except Exception as e:
        return {"params": params, "end_capital": 10000, "max_single_loss": 0, "trades": 0, "error": str(e)}


# -------------------------------
# Optimierung (parallel)
# -------------------------------
def optimize(files, indicator_count=50, depth=1, max_workers=None):
    results = []
    param_sets = list(generate_param_grid(indicator_count, depth))
    print(f"[Optimizer] {len(param_sets)} Parameter-Kombinationen "
          f"für {indicator_count} Indikatoren (Tiefe {depth}).")

    for f in files:
        df = pd.read_parquet(f)
        print(f"[Optimizer] Starte Gridsearch für {f} ...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_backtest_task, (df, p)) for p in param_sets]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parameterkombinationen"):
                results.append(future.result())

    df_results = pd.DataFrame(results)
    df_results.sort_values(["end_capital", "max_single_loss"], ascending=[False, True], inplace=True)

    # Beste Parameter
    best_params = df_results.iloc[0]["params"]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"[Optimizer] Beste Parameter gespeichert: {BEST_PARAMS_PATH}")

    # Ergebnis-Dateien
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_file = f"{RESULTS_DIR}/optimization_results_{timestamp}.xlsx"
    df_results.to_excel(result_file, index=False)
    df_results.to_csv(result_file.replace(".xlsx", ".csv"), index=False)
    df_results.to_markdown(result_file.replace(".xlsx", ".md"))
    print(f"[Optimizer] Ergebnisse gespeichert: {result_file}")

    # Automatische Visualisierung
    try:
        create_all_visuals(result_file)
    except Exception as e:
        print(f"[Optimizer] Visualisierung fehlgeschlagen: {e}")

    return best_params


# -------------------------------
# Main (Interaktiv)
# -------------------------------
def main():
    try:
        indicator_count = int(input("Wie viele Indikatoren sollen optimiert werden? (1-50): ").strip())
    except ValueError:
        indicator_count = 50
    indicator_count = max(1, min(50, indicator_count))

    try:
        depth = int(input("Optimierungstiefe? (1 = schnell, 2 = voll): ").strip())
    except ValueError:
        depth = 1
    depth = 1 if depth != 2 else 2

    files = select_files_dialog()
    if not files:
        print("[Optimizer] Keine Dateien gewählt.")
        return

    best_params = optimize(files, indicator_count, depth)
    print("[Optimizer] Beste Parameter:", best_params)


if __name__ == "__main__":
    main()
