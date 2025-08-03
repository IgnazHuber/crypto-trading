# crypto_trading/analysis/batch_run.py

import argparse
import os
import glob
import pandas as pd
from crypto_trading.analysis.trades import load_asset_data, compute_indicators
from crypto_trading.analysis.strategy import run_strategy
from crypto_trading.analysis.reporting import export_pdf_report, export_csv_reports, export_excel
from crypto_trading.analysis.plots_kpi import (
    plot_equity_curve, plot_drawdown, plot_pnl_hist, plot_month_heatmap,
    plot_portfolio_donut, plot_sharpe_pnl_scatter, plot_pnl_box,
    plot_corr_matrix, plot_tradetimeline
)

DATA_RAW_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
OUTPUT_DIR = "./report_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def list_parquet_files(data_dir=DATA_RAW_DIR):
    files = glob.glob(os.path.join(data_dir, "*.parquet"))
    return sorted(files)

def parse_parquet_filename(fname):
    base = os.path.basename(fname)
    parts = base.replace(".parquet", "").split("_")
    if len(parts) != 5:
        return None
    asset, timeframe, years, _, exchange = parts
    years = float(years.replace("year", ""))
    return {
        "file": fname,
        "asset": asset,
        "timeframe": timeframe,
        "years": years,
        "exchange": exchange
    }

def interactive_parquet_selection():
    files = list_parquet_files()
    if not files:
        raise FileNotFoundError("Keine Parquet-Dateien im data/raw/-Verzeichnis gefunden!")
    print("Verfügbare Parquet-Dateien:")
    options = []
    for i, f in enumerate(files):
        info = parse_parquet_filename(f)
        if info:
            options.append(info)
            print(f"[{i+1}] {info['asset']} | {info['timeframe']} | {info['years']}y | {info['exchange']} | {os.path.basename(f)}")
    inp = input(f"Wähle eine Datei (1-{len(options)}, Enter=Default BTCUSDT_1h_1year_ccxt): ")
    if not inp.strip():
        for idx, info in enumerate(options):
            if (info['asset'], info['timeframe'], info['years'], info['exchange']) == ("BTCUSDT", "1h", 1.0, "ccxt"):
                return info
        print("Default nicht gefunden, nehme erste Datei!")
        return options[0]
    try:
        n = int(inp)
        return options[n-1]
    except Exception:
        print("Ungültige Eingabe, nehme erste Datei!")
        return options[0]

def batch_run(asset, timeframe, years, exchange, config, fast=False):
    print(f"Modus: {'FAST' if fast else 'FULL'}")
    print(f"Starte Analyse für {asset} ({timeframe}, {years}yr, {exchange}) ...")
    df = load_asset_data(asset, timeframe, years, exchange)
    df = compute_indicators(df)
    strat_df = run_strategy(df, config)
    strat_df["Asset"] = asset
    return strat_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="FAST-Flag: Schnell-Modus (nur 1 Asset, kein PDF/Plots)")
    parser.add_argument("--choose", action="store_true", help="Interaktive Auswahl der Parquet-Datei")
    args = parser.parse_args()

    # ==== Parquet-Auswahl ====
    if args.choose:
        sel = interactive_parquet_selection()
        asset, timeframe, years, exchange = sel['asset'], sel['timeframe'], sel['years'], sel['exchange']
    else:
        asset, timeframe, years, exchange = "BTCUSDT", "1h", 1.0, "ccxt"

    # ==== Strategie-Konfiguration ====
    weights = {"MACD": 1.0, "RSI": 0.8, "ADX": 0.5}
    triggers = {"MACD": (0, 999), "RSI": (55, 80), "ADX": (20, 999)}
    config_score = {
        "mode": "score",
        "weights": weights,
        "triggers": triggers,
        "long_threshold": 1.2,
        "short_threshold": -1.2,
    }

    regimes_weights = {
        "trend": {"MACD": 1.0, "EMA_FAST": 0.8, "ADX": 0.7},
        "range": {"RSI": 1.0, "BB_LOWER": 0.8, "BB_UPPER": -0.8},
        "volatile": {"ATR": 1.2, "MACD": 0.5}
    }
    regimes_triggers = {
        "trend": {"MACD": (0, 999), "EMA_FAST": (0, 999), "ADX": (25, 999)},
        "range": {"RSI": (30, 70), "BB_LOWER": (-2, 0), "BB_UPPER": (0, 2)},
        "volatile": {"ATR": (0.8, 999), "MACD": (-1, 1)}
    }
    long_thresholds = {"trend": 1.5, "range": 1.0, "volatile": 1.2}
    short_thresholds = {"trend": -1.5, "range": -1.0, "volatile": -1.2}
    config_adaptive = {
        "mode": "adaptive",
        "regimes_weights": regimes_weights,
        "regimes_triggers": regimes_triggers,
        "long_thresholds": long_thresholds,
        "short_thresholds": short_thresholds,
    }

    # ==== Analyse starten ====
    # Score-Strategie
    results_score = batch_run(asset, timeframe, years, exchange, config_score, fast=args.fast)
    print(results_score.head())

    # Adaptive Strategie
    results_adaptive = batch_run(asset, timeframe, years, exchange, config_adaptive, fast=args.fast)
    print(results_adaptive.head())

    # ==== Export/Reporting ====
    if not args.fast:
        # --- Plots generieren und einbinden ---
        eq = (1 + results_score['score'].fillna(0)).cumprod() if 'score' in results_score.columns else pd.Series([1])
        portfolio_imgs = {
            "Equity Curve": plot_equity_curve(eq, f"{OUTPUT_DIR}/equity_curve.png"),
            "Drawdown": plot_drawdown(eq, f"{OUTPUT_DIR}/drawdown_curve.png"),
            "PnL-Hist": plot_pnl_hist(results_score, f"{OUTPUT_DIR}/pnl_hist.png"),
            "Monats-Heatmap": plot_month_heatmap(results_score, f"{OUTPUT_DIR}/heatmap.png"),
            "Portfolio": plot_portfolio_donut(results_score, f"{OUTPUT_DIR}/portfolio_donut.png"),
            "Sharpe-PnL-Scatter": plot_sharpe_pnl_scatter(results_score, f"{OUTPUT_DIR}/sharpe_pnl_scatter.png"),
            "Outlier-Boxplot": plot_pnl_box(results_score, f"{OUTPUT_DIR}/pnl_box.png"),
            "Corr-Matrix": plot_corr_matrix(results_score, f"{OUTPUT_DIR}/correlation_matrix.png"),
            "Trade-Timeline": plot_tradetimeline(results_score, f"{OUTPUT_DIR}/trade_timeline.png"),
        }
        # --- KPIs (Platzhalter, ggf. durch echte Metriken ersetzen) ---
        kpi_dict = {
            "Sharpe": float(results_score['score'].mean() / results_score['score'].std()) if ('score' in results_score.columns and results_score['score'].std() != 0) else 0,
            "CAGR": 10.0,
            "MaxDrawdown": -12.5,
            "Trefferquote": float((results_score['score'] > 0).mean()) if 'score' in results_score.columns else 0
        }
        asset_param_results = {}  # für spätere Erweiterung

        # --- Exporte ---
        export_csv_reports(results_score, results_score, out_dir=OUTPUT_DIR)
        export_excel(results_score, results_score, out_dir=OUTPUT_DIR)
        export_pdf_report(
            results_score, results_score, asset_param_results, OUTPUT_DIR,
            portfolio_imgs=portfolio_imgs, kpi_dict=kpi_dict,
            FAST_MODE=False, filename=f"{OUTPUT_DIR}/strategy_report.pdf"
        )
    else:
        print("⚡ FAST-Mode: Kein PDF/Plot/Export (nur Analyse/Testing)!")
