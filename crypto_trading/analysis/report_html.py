import pandas as pd

def export_html_summary(df, path):
    """
    Exportiert eine einfache HTML-Datei mit:
    - Überschrift
    - Zusammenfassung der Kennzahlen
    - Parameter-Infos
    """
    html_parts = []

    html_parts.append("<html><head><title>Optimization Summary</title></head><body>")
    html_parts.append("<h1>Optimization Summary</h1>")

    # Summary-Kennzahlen
    html_parts.append("<h2>Kennzahlen</h2>")
    html_parts.append(df.describe(include="all").to_html())

    # Parameterdetails (falls vorhanden)
    if "params" in df.columns:
        html_parts.append("<h2>Parameter-Details</h2>")
        params_table = []
        for i, row in df.iterrows():
            params = row["params"] if isinstance(row["params"], dict) else {}
            row_html = f"<tr><td>{i}</td>" + "".join(f"<td>{k}: {v}</td>" for k, v in params.items()) + "</tr>"
            params_table.append(row_html)
        html_parts.append("<table border=1><tr><th>Index</th><th>Parameter</th></tr>")
        html_parts.extend(params_table)
        html_parts.append("</table>")
    else:
        html_parts.append("<p>Keine Parameterinformationen verfügbar</p>")

    html_parts.append("</body></html>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"[HTML] Exportiert: {path}")
