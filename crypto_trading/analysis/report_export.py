import pandas as pd

def export_excel_summary(df, end_capital=None, max_loss=None, path="results/optimization_summary.xlsx"):
    """
    Speichert Optimierungsergebnisse und optional eine Zusammenfassung in Excel.
    """
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="OptimizationResults", index=False)
        if end_capital is not None and max_loss is not None:
            summary = pd.DataFrame([{"Endkapital": end_capital, "MaxVerlust": max_loss}])
            summary.to_excel(writer, sheet_name="Summary", index=False)
    print(f"[Report] Excel exportiert: {path}")
