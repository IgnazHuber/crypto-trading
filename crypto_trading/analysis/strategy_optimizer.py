import os
import sys
import json
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from tqdm import tqdm
import webbrowser
from datetime import datetime

from .indicator_library import calc_all_indicators
from .indicator_context import classify_trend, classify_volatility, classify_volume, apply_context_to_indicators
from .mod_trades import generate_signals
from .optimizer_visuals_ext import create_visuals

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
BEST_PARAMS_FILE = os.path.join(RESULTS_DIR, "best_params.json")

INDICATOR_PRIORITY = [
    "ema", "sma", "rsi", "macd", "mfi", "atr", "bbands", "stoch", "stochrsi",
    "williams_r", "ultimate_osc", "awesome_osc", "roc", "mom", "obv", "cmf",
    "tema", "dema", "hma", "wma", "ppo", "kc", "vortex", "dmi", "trix", "cci",
    "eom", "adx", "vwap", "ha_close", "ha_open", "bullp", "bearp", "cmf_20",
    "eom", "tema_20", "dema_20", "hma_20", "wma_20", "ppo_12_26_9", "ppoh",
    "ppos", "kcle", "kcbe", "kcue", "vtxp", "vtxm", "dmp", "dmn"
]

def select_files():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilenames(
        title="Wähle Parquet-Dateien",
        initialdir=r"D:\Projekte\crypto_trading\crypto_trading\data\raw",
        filetypes=[("Parquet Dateien", "*.parquet")]
    )

def split_by_market_context(df):
    df = calc_all_indicators(df.copy(), {})
    try:
        df = classify_trend(df)
        df = classify_volatility(df)
        df = classify_volume(df)
        df = apply_context_to_indicators(df)
        contexts = df["market_context"].unique()
        print(f"[Context] Marktumfelder erkannt: {list(contexts)}")
    except Exception as e:
        print(f"[WARN] Kontextklassifizierung fehlgeschlagen ({e}) – nutze 'sideways_normal_volatility_normal_volume'.")
        df["market_context"] = "sideways_normal_volatility_normal_volume"
    return {ctx: df[df["market_context"] == ctx].copy() for ctx in df["market_context"].unique()}

def simulate_trades_with_capital(df, params):
    capital = 10000.0
    max_loss = 0.0
    signals = generate_signals(df, params)
    for sig in signals:
        trade_size = capital * 0.10
        raw_pnl = sig.get("pnl_abs", 0.0)
        pnl = max(min(raw_pnl, trade_size * 0.20), -trade_size * 0.03)
        capital += pnl
        max_loss = min(max_loss, pnl)
    return capital, max_loss

def optimize_for_context(df_ctx, param_grid):
    results = []
    for params in tqdm(param_grid, desc=f"{df_ctx['market_context'].iloc[0]} Parameter"):
        end_capital, max_loss = simulate_trades_with_capital(df_ctx, params)
        results.append({
            "context": df_ctx["market_context"].iloc[0],
            "end_capital": end_capital,
            "max_single_loss": max_loss,
            "params": params
        })
    return pd.DataFrame(results)

def create_param_grid(n_indicators):
    grid = []
    top_indicators = INDICATOR_PRIORITY[:n_indicators]
    for ema_len in ([10, 20, 50, 100] if "ema" in top_indicators else [20]):
        for sma_len in ([20, 50, 100, 200] if "sma" in top_indicators else [50]):
            for rsi_len in ([7, 14, 28] if "rsi" in top_indicators else [14]):
                for atr_len in ([7, 14] if "atr" in top_indicators else [14]):
                    for stoch_k in ([14] if "stoch" in top_indicators else [14]):
                        for stoch_d in ([3] if "stoch" in top_indicators else [3]):
                            for macd_fast in ([8, 12] if "macd" in top_indicators else [12]):
                                for macd_slow in ([17, 26] if "macd" in top_indicators else [26]):
                                    params = {
                                        "ema_length": ema_len,
                                        "sma_length": sma_len,
                                        "rsi_length": rsi_len,
                                        "atr_length": atr_len,
                                        "stoch_k": stoch_k,
                                        "stoch_d": stoch_d,
                                        "macd_fast": macd_fast,
                                        "macd_slow": macd_slow,
                                        "macd_signal": 9
                                    }
                                    grid.append(params)
    return grid

def main():
    file_paths = select_files()
    if not file_paths:
        print("[Optimizer] Keine Dateien ausgewählt – Abbruch.")
        return
    try:
        n_indicators = int(input("Wie viele Indikatoren sollen optimiert werden? (1-50): "))
    except ValueError:
        n_indicators = 10
    print(f"[Optimizer] Optimierung für {n_indicators} Indikatoren")

    param_grid = create_param_grid(n_indicators)
    all_results = []
    best_params_by_context = {}

    for file_path in file_paths:
        print(f"[Optimizer] Lade Datei: {file_path}")
        df = pd.read_parquet(file_path)
        df_splits = split_by_market_context(df)
        for ctx, df_ctx in df_splits.items():
            if df_ctx.empty:
                print(f"[Optimizer] Keine Daten für {ctx}")
                continue
            res = optimize_for_context(df_ctx, param_grid)
            all_results.append(res)
            best = res.sort_values("end_capital", ascending=False).iloc[0]
            best_params_by_context[ctx] = best["params"]

    all_results_df = pd.concat(all_results, ignore_index=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    excel_path = os.path.join(RESULTS_DIR, f"optimization_results_by_context_{timestamp}.xlsx")
    all_results_df.to_excel(excel_path, index=False)
    print(f"[Optimizer] Ergebnisse gespeichert: {excel_path}")

    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(best_params_by_context, f, indent=2)
    print(f"[Optimizer] Beste Parameter je Marktumfeld gespeichert: {BEST_PARAMS_FILE}")

    html_path = os.path.join(RESULTS_DIR, f"optimization_results_by_context_{timestamp}_visuals.html")
    create_visuals(all_results_df, html_path)
    webbrowser.open(f"file://{os.path.abspath(html_path)}")

if __name__ == "__main__":
    main()
