import itertools
import pandas as pd
from crypto_trading.trades import generate_all_trades
from crypto_trading.visualization.pdf_report import create_pdf_report
from crypto_trading.indicator_legend import get_indicator_legend

def export_trades_to_excel(trades_df, excel_path, indicator_weights=None):
    df = trades_df.copy()
    if indicator_weights is not None:
        df['Gewichtungen'] = str(indicator_weights)
    if 'analysis_short' not in df.columns:
        df['analysis_short'] = "-"
    if 'analysis_long' not in df.columns:
        df['analysis_long'] = "-"
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(1)
    df.to_excel(excel_path, index=False)
    print(f"Excel erfolgreich exportiert: {excel_path}")

WEIGHT_SPACE = {
    "MACD": [1, 2, 3],
    "ADX": [1, 2],
    "RSI_14": [1, 2],
    "BB_LOWER": [0, 1],
    "VOLUME_SMA": [0, 1],
}

ENTRY_THRESHOLDS = [1, 2, 3]
EXIT_THRESHOLDS = [0, 1, 2]

def calc_performance(trades_df):
    import numpy as np
    returns = trades_df["pnl_pct"] / 100 if not trades_df.empty else pd.Series(dtype=float)
    if returns.empty:
        return {"Sharpe": 0, "MaxDrawdown": 0, "CAGR": 0}
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = (mean_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
    equity = (1 + returns).cumprod()
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = drawdown.min()
    n_years = (trades_df["exit_date"].max() - trades_df["entry_date"].min()).days / 365.25 if len(trades_df) > 0 else 0
    cagr = equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    hit_rate = 100 * (trades_df["pnl_abs"] > 0).mean() if len(trades_df) > 0 else 0
    win_ratio = (
        abs(trades_df[trades_df["pnl_abs"] > 0]["pnl_abs"].mean() /
            trades_df[trades_df["pnl_abs"] < 0]["pnl_abs"].mean())
        if (trades_df["pnl_abs"] < 0).any() else "-"
    )
    return {
        "Sharpe": round(sharpe, 2),
        "MaxDrawdown": f"{max_dd*100:.2f}%",
        "CAGR": f"{cagr*100:.2f}%",
        "Trefferquote (%)": f"{hit_rate:.1f}",
        "Gewinn/Verlust-Ratio": f"{win_ratio:.2f}" if win_ratio != "-" else "-"
    }

def grid_search_optimizer():
    all_results = []
    keys = list(WEIGHT_SPACE.keys())
    values = [WEIGHT_SPACE[k] for k in keys]

    for w_combo in itertools.product(*values):
        weights = dict(zip(keys, w_combo))
        for entry_thr in ENTRY_THRESHOLDS:
            for exit_thr in EXIT_THRESHOLDS:
                trades_df, price_data = generate_all_trades(
                    time_range="12m",
                    entry_threshold=entry_thr,
                    exit_threshold=exit_thr,
                    weights=weights
                )
                if len(trades_df) < 3:
                    continue
                performance = calc_performance(trades_df)
                all_results.append({
                    **weights,
                    "entry_threshold": entry_thr,
                    "exit_threshold": exit_thr,
                    "n_trades": len(trades_df),
                    "sharpe": performance["Sharpe"],
                    "max_dd": performance["MaxDrawdown"],
                    "cagr": performance["CAGR"],
                    "mean_pnl": trades_df["pnl_abs"].mean(),
                })

    df_results = pd.DataFrame(all_results)
    if df_results.empty or "sharpe" not in df_results.columns:
        print("WARNUNG: Keine Ergebnisse/Spalte 'sharpe' vorhanden! Prüfe Parameter und Daten.")
        return df_results
    df_results = df_results.sort_values(by="sharpe", ascending=False)
    return df_results

def run_optimizer_and_export(top_n=3):
    results = grid_search_optimizer()
    if results.empty:
        print("Abbruch: Keine Strategiekombination mit mindestens 3 Trades gefunden.")
        return

    results.to_excel("results/strategy_grid_search.xlsx", index=False)
    print("Grid-Search-Resultate gespeichert: results/strategy_grid_search.xlsx")

    indicator_legend = get_indicator_legend()

    for i in range(min(top_n, len(results))):
        params = results.iloc[i]
        weights = {k: params[k] for k in WEIGHT_SPACE.keys()}
        entry_thr = params["entry_threshold"]
        exit_thr = params["exit_threshold"]

        trades_df, price_data = generate_all_trades(
            time_range="12m",
            entry_threshold=entry_thr,
            exit_threshold=exit_thr,
            weights=weights
        )
        portfolio_summary = calc_performance(trades_df)
        pdf_path = f"results/strategy_top_{i+1}.pdf"
        charts_dir = "analysis_results/charts"
        create_pdf_report(
            trades_df,
            price_data,
            portfolio_summary,
            pdf_path,
            charts_dir,
            indicator_legend,
            indicator_weights=weights
        )
        excel_path = f"results/strategy_top_{i+1}.xlsx"
        export_trades_to_excel(trades_df, excel_path, indicator_weights=weights)
        print(f"PDF-Report und Excel für Top-{i+1} exportiert: {pdf_path}, {excel_path}")

if __name__ == "__main__":
    run_optimizer_and_export(top_n=3)
