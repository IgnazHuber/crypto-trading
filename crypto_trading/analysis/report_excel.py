import pandas as pd
import xlsxwriter

def export_excel_summary(df, path):
    """
    Exportiert eine Excel-Datei mit mehreren Tabs:
    - Summary: Kennzahlen
    - All_Trades: alle Trades aus df
    - Indicators_Used: Liste aller Indikatoren & Parameter
    - Charts: Platzhalter für spätere Diagramme
    """
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        # --- Summary ---
        summary = pd.DataFrame({
            "Kennzahl": [
                "Zeilen",
                "Durchschnittliches Kapital",
                "Max Verlust (Trade)"
            ],
            "Wert": [
                len(df),
                df["end_capital"].mean() if "end_capital" in df.columns else None,
                df["max_single_loss"].min() if "max_single_loss" in df.columns else None
            ]
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)

        # --- All Trades ---
        df.to_excel(writer, sheet_name="All_Trades", index=False)

        # --- Indicators Used ---
        if "params" in df.columns:
            # Parameter dicts aufbereiten
            params_list = []
            for i, row in df.iterrows():
                params = row["params"] if isinstance(row["params"], dict) else {}
                for k, v in params.items():
                    params_list.append({"Index": i, "Parameter": k, "Wert": v})
            pd.DataFrame(params_list).to_excel(writer, sheet_name="Indicators_Used", index=False)
        else:
            pd.DataFrame([{"Info": "Keine Parameterinformationen"}]).to_excel(
                writer, sheet_name="Indicators_Used", index=False
            )

        # --- Charts (Platzhalter) ---
        chart_placeholder = pd.DataFrame([{"Hinweis": "Hier können Charts eingefügt werden"}])
        chart_placeholder.to_excel(writer, sheet_name="Charts", index=False)

        # --- Formatierung (optional) ---
        workbook = writer.book
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.autofilter(0, 0, worksheet.dim_rowmax, worksheet.dim_colmax)

    print(f"[Excel] Mehrtabbiger Export erzeugt: {path}")
