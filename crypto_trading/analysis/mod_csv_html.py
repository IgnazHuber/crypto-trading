# mod_csv_html.py
import pandas as pd

def export_to_csv_and_html(summary_df, all_trades_df,
                           params=None, indicator_info=None,
                           summary_csv="results/trades_summary_all.csv",
                           summary_html="results/trades_summary_all.html",
                           trades_html="results/all_trades.html"):
    # CSV
    summary_df.to_csv(summary_csv, index=False)
    all_trades_df.to_csv("results/all_trades.csv", index=False)

    # HTML Summary
    html_parts = []
    if params:
        html_parts.append("<h2>Verwendete Parameter</h2><ul>")
        for k, v in params.items():
            html_parts.append(f"<li><b>{k}</b>: {v}</li>")
        html_parts.append("</ul>")
    if indicator_info:
        html_parts.append("<h2>Indikatoren & Marktumfeld</h2><ul>")
        for k, info in indicator_info.items():
            html_parts.append(
                f"<li><b>{info.get('title', k)}</b> – {info.get('context','')}<br>"
                + "<br>".join(info.get("bullets", [])) + "</li>")
        html_parts.append("</ul>")
    html_parts.append("<h2>Summary</h2>")
    html_parts.append(summary_df.to_html(index=False))
    with open(summary_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    # HTML Trades separat
    if not all_trades_df.empty:
        trades_html_code = "<h2>Alle Trades</h2>" + all_trades_df.to_html(index=False)
        with open(trades_html, "w", encoding="utf-8") as f:
            f.write(trades_html_code)
