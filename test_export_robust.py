import os
import pandas as pd
import numpy as np
from crypto_trading.analysis.reporting import export_csv_reports, export_excel, export_pdf_report
from crypto_trading.analysis.plots_kpi import (
    plot_equity_curve, plot_drawdown, plot_pnl_hist, plot_month_heatmap,
    plot_portfolio_donut, plot_sharpe_pnl_scatter, plot_pnl_box,
    plot_corr_matrix, plot_tradetimeline
)

OUTPUT_DIR = "./report_output_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Leerer DataFrame: ---
df_empty = pd.DataFrame()

# --- Teilweise befüllter DataFrame (nur eine Zeile, viele NaN, falsche Spaltennamen): ---
df_partial = pd.DataFrame({
    "open": [np.nan], "high": [np.nan], "close": [np.nan], "score": [np.nan],
    "irgendwas": ["foo"], "entry_time": [""], "gewinn_verlust": [np.nan], "asset": [None]
})

# --- Ein „fast korrekter“ DataFrame, aber mit einer Spalte nur Strings: ---
df_almost = pd.DataFrame({
    "entry_time": ["2024-01-01 12:00"],
    "gewinn_verlust": ["keine Zahl"],
    "score": [0.0],
    "asset": ["BTCUSDT"]
})

# --- Ein Minimalbeispiel, wie von einer echten Strategie: ---
df_real = pd.DataFrame({
    "entry_time": pd.date_range("2024-01-01", periods=5, freq="D"),
    "gewinn_verlust": [1.1, -0.4, 2.0, -3.2, 0.0],
    "score": [0.1, -0.2, 0.05, -0.1, 0.0],
    "asset": ["BTCUSDT"] * 5
})

test_cases = [
    ("Empty DataFrame", df_empty),
    ("Partial DataFrame", df_partial),
    ("Almost Correct DataFrame", df_almost),
    ("Realistic DataFrame", df_real),
]

for name, df in test_cases:
    print(f"\n===== Testfall: {name} =====")
    # Plots
    eq = (1 + pd.to_numeric(df.get('score', pd.Series()), errors="coerce").fillna(0)).cumprod() if 'score' in df else pd.Series([1])
    plot_files = {
        "Equity Curve": plot_equity_curve(eq, f"{OUTPUT_DIR}/equity_curve_{name}.png"),
        "Drawdown": plot_drawdown(eq, f"{OUTPUT_DIR}/drawdown_{name}.png"),
        "PnL-Hist": plot_pnl_hist(df, f"{OUTPUT_DIR}/pnl_hist_{name}.png"),
        "Monats-Heatmap": plot_month_heatmap(df, f"{OUTPUT_DIR}/heatmap_{name}.png"),
        "Portfolio": plot_portfolio_donut(df, f"{OUTPUT_DIR}/portfolio_donut_{name}.png"),
        "Sharpe-PnL-Scatter": plot_sharpe_pnl_scatter(df, f"{OUTPUT_DIR}/sharpe_pnl_scatter_{name}.png"),
        "Outlier-Boxplot": plot_pnl_box(df, f"{OUTPUT_DIR}/pnl_box_{name}.png"),
        "Corr-Matrix": plot_corr_matrix(df, f"{OUTPUT_DIR}/correlation_matrix_{name}.png"),
        "Trade-Timeline": plot_tradetimeline(df, f"{OUTPUT_DIR}/trade_timeline_{name}.png"),
    }
    # KPIs robust, ggf. Dummy
    kpi_dict = {
        "Sharpe": float(pd.to_numeric(df.get('score', pd.Series()), errors="coerce").mean() /
                        pd.to_numeric(df.get('score', pd.Series()), errors="coerce").std()) if 'score' in df and pd.to_numeric(df.get('score', pd.Series()), errors="coerce").std() != 0 else 0,
        "CAGR": 0.0,
        "MaxDrawdown": 0.0,
        "Trefferquote": float((pd.to_numeric(df.get('score', pd.Series()), errors="coerce") > 0).mean()) if 'score' in df else 0
    }
    asset_param_results = {}

    # Exports
    export_csv_reports(df, df, out_dir=OUTPUT_DIR)
    export_excel(df, df, out_dir=OUTPUT_DIR)
    export_pdf_report(
        df, df, asset_param_results, OUTPUT_DIR,
        portfolio_imgs=plot_files, kpi_dict=kpi_dict,
        FAST_MODE=False, filename=f"{OUTPUT_DIR}/strategy_report_{name}.pdf"
    )

print("\nAlle Testfälle durchlaufen. Prüfe report_output_test/ auf erzeugte Dateien.")
