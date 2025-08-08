# reporting/export_excel.py

import pandas as pd

def export_kpis_and_trades(kpi_list, all_trades, output_excel, sheet_names=None):
    """
    Exportiert KPIs und alle Einzeltrades in eine Excel-Datei mit mehreren Sheets.
    Alle Werte (außer Trade-ID) werden auf 1 Nachkommastelle gerundet.
    
    Args:
        kpi_list:   Liste von KPI-Dictionaries für alle Strategien.
        all_trades: Liste von DataFrames mit allen Einzeltrades pro Strategie.
        output_excel: Zielpfad der Excel-Datei.
        sheet_names: Optional: Sheet-Namen für jede Strategie (List[str]).
    """
    # KPIs DataFrame
    perf_df = pd.DataFrame(kpi_list)
    # Einzeltrades DataFrame
    all_trades_df = pd.concat(all_trades, ignore_index=True)
    # Formatierung: Runde alle Werte (außer Trade-ID) auf 1 Nachkommastelle
    for col in all_trades_df.columns:
        if col != "trade_id" and all_trades_df[col].dtype.kind in "fi":
            all_trades_df[col] = all_trades_df[col].round(1)
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        perf_df.to_excel(writer, sheet_name="KPIs", index=False)
        all_trades_df.to_excel(writer, sheet_name="Alle Trades", index=False)
        # Optionale Einzel-Sheets für jede Strategie
        if sheet_names is not None:
            for trades, strat in zip(all_trades, sheet_names):
                # Auch hier auf 1 Nachkommastelle runden (außer trade_id)
                trades_ = trades.copy()
                for col in trades_.columns:
                    if col != "trade_id" and trades_[col].dtype.kind in "fi":
                        trades_[col] = trades_[col].round(1)
                trades_.to_excel(writer, sheet_name=strat, index=False)
    print(f"[Excel] Exportiert: {output_excel}")
