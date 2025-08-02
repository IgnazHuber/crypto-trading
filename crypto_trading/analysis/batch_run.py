"""
batch_run.py

Zentrale Steuerung des gesamten Analyse-Workflows:
- Parquet-Auswahl, Param-Grid/FAST-Flag
- Batch- und Grid-Loop, Parallelisierung (Multi-Core)
- Trade- und KPI-Berechnung je Asset/Paramset (strategy.py, kpi.py)
- Alle Visuals & Wissenschaftsplots (plots_kpi.py)
- Reporting (PDF, CSV), Fortschrittsbalken, Logging

Benötigt:
- strategy.py, kpi.py, plots_kpi.py, reporting.py, utils.py (im PYTHONPATH, z. B. analysis/)
- Parquet-Daten im Data-Pfad

Author: ChatGPT Research, 2025
"""

import os
import tempfile
import shutil
import concurrent.futures
import pandas as pd
import numpy as np

from analysis.strategy import run_score_strategy
from analysis.kpi import performance_metrics, extended_kpis, portfolio_kpis
from analysis.plots_kpi import (
    plot_equity_curve, plot_drawdown, plot_rolling_sharpe, plot_pnl_hist,
    plot_month_heatmap, plot_sharpe_pnl_scatter, plot_portfolio_donut,
    plot_pnl_box, plot_corr_matrix, plot_tradetimeline
)
from analysis.reporting import export_pdf_report, export_csv_reports
from analysis.utils import (
    list_parquet_files, select_parquet_files, load_parquet_file,
    build_param_grid, progress_bar, unique_run_dir, logger
)

# ======= Einstellungen =======
DATA_PATH = r"c:\Projekte\crypto_trading\crypto_trading\data\raw"
EXPORT_PDF = True
EXPORT_CSV = True
FAST_ANALYSIS = False    # Auf True für schnellen Test!
N_TOP = 5                # Top/Flop-Trades pro Asset/Paramset
MAX_WORKERS = os.cpu_count() or 4

# === Parameter-Grid ===
GRID = {
    "RSI_PERIOD": [8, 14, 21],
    "MACD_FAST": [8, 12],
    "MACD_SLOW": [18, 26],
    "MACD_SIGNAL": [6, 9],
    "EMA_SHORT": [21, 50],
    "EMA_LONG": [100, 200],
    "BB_WINDOW": [10, 20],
    "STOCH_WINDOW": [8, 14],
    "STOP_LOSS_PCT": [0.02, 0.03, 0.05],
    "TAKE_PROFIT_PCT": [0.04, 0.06, 0.1],
}
DEFAULT_FILES = ["BTCUSDT_1h_1year_ccxt.parquet"]

def process_asset_param(args):
    """
    Läuft je Asset/Paramset als Worker (auch für Parallelisierung).
    Gibt dict mit allen Daten, Plots, KPIs zurück.
    """
    parquet_file, params, param_idx, param_names, asset, tmpdir = args
    df = load_parquet_file(parquet_file)
    trades = run_score_strategy(
        df, params, asset,
        stop_loss_pct=params[-2],
        take_profit_pct=params[-1],
        trade_mode='both'
    )
    # Equity
    eq = pd.Series(1.0, index=pd.to_datetime(trades['exit_time']))
    for idx, trade in trades.iterrows():
        pnl = trade['gewinn_verlust'] / 100.0
        eq.iloc[idx:] = eq.iloc[idx:] * (1 + pnl)
    eq = eq.dropna()
    # KPIs
    perf = performance_metrics(trades, params, asset, param_names)
    # Erweiterte KPIs (Rolling, Drawdown, etc.)
    ext_kpis = extended_kpis(trades, eq)
    # Plots/Visuals
    eqfile = os.path.join(tmpdir, f"{asset}_{param_idx}_eq.png")
    drfile = os.path.join(tmpdir, f"{asset}_{param_idx}_drawdown.png")
    shfile = os.path.join(tmpdir, f"{asset}_{param_idx}_rolling_sharpe.png")
    monfile = os.path.join(tmpdir, f"{asset}_{param_idx}_monat.png")
    histfile = os.path.join(tmpdir, f"{asset}_{param_idx}_hist.png")
    boxfile = os.path.join(tmpdir, f"{asset}_{param_idx}_box.png")
    heatmapfile = os.path.join(tmpdir, f"{asset}_{param_idx}_heatmap.png")
    timelinefile = os.path.join(tmpdir, f"{asset}_{param_idx}_timeline.png")

    plot_equity_curve(eq, eqfile)
    plot_drawdown(eq, drfile)
    plot_rolling_sharpe(eq, shfile)
    plot_month_heatmap(trades, monfile)
    plot_pnl_hist(trades, histfile)
    plot_pnl_box(trades, boxfile)
    plot_month_heatmap(trades, heatmapfile)
    plot_tradetimeline(trades, timelinefile)

    return {
        'asset': asset, 'param_idx': param_idx, 'params': params,
        'trades': trades, 'perf': perf, 'eqfile': eqfile, 'drfile': drfile,
        'shfile': shfile, 'monfile': monfile, 'histfile': histfile,
        'boxfile': boxfile, 'heatmapfile': heatmapfile, 'timelinefile': timelinefile,
        'ext_kpis': ext_kpis
    }

def main():
    # === Parquet-Dateien wählen ===
    files = list_parquet_files(DATA_PATH)
    parquet_files = select_parquet_files(files, DEFAULT_FILES)
    parquet_files = [os.path.join(DATA_PATH, f) for f in parquet_files]

    # === Param-Grid bauen ===
    param_grid, param_names = build_param_grid(GRID, FAST_ANALYSIS)

    # === Vorbereitung Batch & Reporting ===
    tmpdir = tempfile.mkdtemp()
    run_dir = unique_run_dir("./runs")

    logger(f"Starte Analyse: {len(parquet_files)} Assets x {len(param_grid)} Paramsets, FAST={FAST_ANALYSIS}")
    tasks = []
    asset_names = []
    for parquet_file in parquet_files:
        asset = os.path.basename(parquet_file).split("_")[0]
        asset_names.append(asset)
        for param_idx, params in enumerate(param_grid):
            tasks.append((parquet_file, params, param_idx, param_names, asset, tmpdir))

    all_trades, all_results, asset_param_results = [], [], {asset: {} for asset in asset_names}

    # === Parallel-Execution Batch (mit Fortschrittsbalken) ===
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futs = [executor.submit(process_asset_param, t) for t in tasks]
        for fut in progress_bar(concurrent.futures.as_completed(futs), total=len(futs), desc="Batch-Analyse"):
            result = fut.result()
            trades = result['trades']
            perf = result['perf']
            all_trades.append(trades)
            all_results.append(perf)
            asset_param_results[result['asset']][result['param_idx']] = {
                'trades': trades, 'perf': perf,
                'eqfile': result['eqfile'],
                'drfile': result['drfile'],
                'shfile': result['shfile'],
                'monfile': result['monfile'],
                'histfile': result['histfile'],
                'boxfile': result['boxfile'],
                'heatmapfile': result['heatmapfile'],
                'timelinefile': result['timelinefile'],
                'ext_kpis': result['ext_kpis']
            }

    # === Ergebnis-DataFrames ===
    df_all_trades = pd.concat(all_trades, ignore_index=True)
    df_perf = pd.DataFrame(all_results)

    # === Portfolio-Gesamtkurven und Visuals ===
    port_eq = pd.Series(1.0)
    df_all_trades = df_all_trades.sort_values('exit_time')
    for idx, trade in df_all_trades.iterrows():
        pnl = trade['gewinn_verlust'] / 100.0
        port_eq = port_eq.append(pd.Series(port_eq.iloc[-1] * (1 + pnl), index=[pd.to_datetime(trade['exit_time'])]))
    port_eq = port_eq[~port_eq.index.duplicated()]
    portfolio_imgs = {}
    port_eqfile = os.path.join(tmpdir, "portfolio_equity.png")
    port_drfile = os.path.join(tmpdir, "portfolio_drawdown.png")
    port_monfile = os.path.join(tmpdir, "portfolio_monat.png")
    port_donutfile = os.path.join(tmpdir, "portfolio_donut.png")
    port_boxfile = os.path.join(tmpdir, "portfolio_box.png")
    port_scatterfile = os.path.join(tmpdir, "portfolio_sharpe_pnl.png")
    port_corrfile = os.path.join(tmpdir, "portfolio_corr.png")
    port_timelinefile = os.path.join(tmpdir, "portfolio_timeline.png")

    plot_equity_curve(port_eq, port_eqfile, title="Portfolio Equity Curve (Alle Trades/Assets)")
    plot_drawdown(port_eq, port_drfile)
    plot_month_heatmap(df_all_trades, port_monfile, title="Portfolio Monatsrenditen (Alle Trades/Assets)")
    plot_portfolio_donut(df_all_trades, port_donutfile)
    plot_pnl_box(df_all_trades, port_boxfile)
    plot_sharpe_pnl_scatter(df_perf, port_scatterfile)
    plot_corr_matrix(df_perf, port_corrfile)
    plot_tradetimeline(df_all_trades, port_timelinefile)

    portfolio_imgs = {
        "Portfolio Equity Curve": port_eqfile,
        "Portfolio Drawdown": port_drfile,
        "Monatsrenditen": port_monfile,
        "Portfolio Allokation": port_donutfile,
        "Outlier/Boxplot": port_boxfile,
        "Sharpe-PnL-Scatter": port_scatterfile,
        "Korrelation": port_corrfile,
        "Trade-Timeline": port_timelinefile
    }
    # === Portfolio-KPIs ===
    kpi_dict = portfolio_kpis(df_all_trades, eq_curve=port_eq)

    # === Reporting ===
    if EXPORT_CSV:
        export_csv_reports(df_all_trades, df_perf, out_dir=run_dir)
    if EXPORT_PDF:
        export_pdf_report(
            df_all_trades, df_perf, asset_param_results, tmpdir,
            portfolio_imgs=portfolio_imgs, kpi_dict=kpi_dict, FAST_MODE=FAST_ANALYSIS,
            filename=os.path.join(run_dir, "strategy_report.pdf")
        )

    # === Cleanup ===
    shutil.rmtree(tmpdir)
    logger(f"Fertig. Alle Reports liegen unter: {run_dir}")

if __name__ == "__main__":
    main()
