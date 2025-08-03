# crypto_trading/analysis/export_utils.py

import os
import pandas as pd

def export_csv_and_excel(
    df_all_trades, df_perf, out_dir="./report_output",
    doku_trades=None, doku_perf=None
):
    """
    Exportiert alle Trades und KPIs als CSV und Excel.
    Excel-Sheet enthält: Trades, Performance, Dokumentation aller Felder.
    """
    os.makedirs(out_dir, exist_ok=True)
    # --- CSV ---
    df_all_trades.to_csv(os.path.join(out_dir, "trades_detailed.csv"), index=False)
    df_perf.to_csv(os.path.join(out_dir, "strategy_performance_grid.csv"), index=False)

    # --- Doku-DataFrames automatisch generieren, falls nicht vorhanden
    if doku_trades is None:
        doku_trades = pd.DataFrame({
            "Spalte": list(df_all_trades.columns),
            "Beispiel": [str(df_all_trades.iloc[0][c]) if len(df_all_trades)>0 else "" for c in df_all_trades.columns],
            "Bedeutung": [
                "Asset Name", "Parameter-Index", "Trade-Nummer", "Kaufpreis", "Verkaufspreis",
                "Absoluter/prozentualer Gewinn/Verlust",
                "Kurzanalyse", "Begründung", "Verbesserungsvorschlag",
                "Score", "Kaufzeit", "Verkaufszeit", "Richtung (Long/Short/Flat)"
            ][:len(df_all_trades.columns)]
        })
    if doku_perf is None:
        doku_perf = pd.DataFrame({
            "Spalte": list(df_perf.columns),
            "Bedeutung": list(df_perf.columns)  # Hier ggf. eigene Erklärungen ergänzen
        })

    # --- Excel ---
    excel_path = os.path.join(out_dir, "full_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_all_trades.to_excel(writer, sheet_name="Trades", index=False)
        df_perf.to_excel(writer, sheet_name="Performance", index=False)
        doku_trades.to_excel(writer, sheet_name="Doku_Trades", index=False)
        doku_perf.to_excel(writer, sheet_name="Doku_Performance", index=False)
    print(f"CSV & Excel-Export: {out_dir}")
