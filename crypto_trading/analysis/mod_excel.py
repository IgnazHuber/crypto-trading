# mod_excel.py
import pandas as pd
from .indicator_descriptions import INDICATOR_INFO

def export_to_excel(summary_df, all_trades_df, equity_dict, params=None,
                    path="results/trades_summary_all.xlsx"):
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        # Summary
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # All Trades
        all_trades_df.to_excel(writer, sheet_name="All_Trades", index=False)

        workbook = writer.book
        # Einzel-Equity Tabs mit Diagrammen
        for name, df in equity_dict.items():
            sheet_name = (name[:28] + "_" + str(len(df)))[:31]  # eindeutige, gültige Namen
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            if len(df) > 1:
                chart = workbook.add_chart({"type": "line"})
                chart.add_series({
                    "name":       "Equity",
                    "categories": [sheet_name, 1, 0, len(df), 0],
                    "values":     [sheet_name, 1, 1, len(df), 1],
                })
                chart.add_series({
                    "name":       "Drawdown",
                    "categories": [sheet_name, 1, 0, len(df), 0],
                    "values":     [sheet_name, 1, 2, len(df), 2],
                    "y2_axis":    True,
                })
                chart.set_title({"name": f"Equity & Drawdown - {name}"})
                chart.set_x_axis({"name": "Time"})
                chart.set_y_axis({"name": "Equity"})
                chart.set_y2_axis({"name": "Drawdown"})
                writer.sheets[sheet_name].insert_chart("E2", chart)

        # === Indikator-Tabelle ===
        if params:
            rows = []
            for k, v in params.items():
                info = INDICATOR_INFO.get(k, {})
                bullets = "\n".join(info.get("bullets", []))
                rows.append([info.get("title", k), v, info.get("context", ""), bullets])
            df_ind = pd.DataFrame(rows, columns=["Indikator","Parameter","Marktumfeld","Beschreibung"])
            df_ind.to_excel(writer, sheet_name="Indicators_Used", index=False)

    print(f"[Excel] Export abgeschlossen: {path}")
