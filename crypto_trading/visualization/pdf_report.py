from fpdf import FPDF
import os

FONT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fonts/DejaVuSans.ttf"))
FONT_PATH_BOLD = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fonts/DejaVuSans-Bold.ttf"))
FONT_PATH_ITALIC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fonts/DejaVuSans-Oblique.ttf"))

def fmt_number(val):
    if isinstance(val, float):
        as_str = f"{val:.1f}"
        return as_str if ".0" not in as_str else as_str.replace(".0", "")
    return str(val)

def create_pdf_report(
    trades_df, price_data, portfolio_summary, pdf_path, charts_dir,
    indicator_legend_full, indicator_weights=None
):
    for fp, name in [
        (FONT_PATH, "DejaVuSans.ttf"),
        (FONT_PATH_BOLD, "DejaVuSans-Bold.ttf"),
        (FONT_PATH_ITALIC, "DejaVuSans-Oblique.ttf"),
    ]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Font-Datei nicht gefunden: {fp} (erwartet als {name})")

    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_font('DejaVu', '', FONT_PATH, uni=True)
    pdf.add_font('DejaVu', 'B', FONT_PATH_BOLD, uni=True)
    pdf.add_font('DejaVu', 'I', FONT_PATH_ITALIC, uni=True)

    pdf.set_font('DejaVu', 'B', 16)
    pdf.add_page()
    pdf.cell(0, 10, 'Trading Strategy Report', ln=True)
    pdf.set_font('DejaVu', '', 12)
    for k, v in portfolio_summary.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)
    pdf.ln(4)

    pdf.set_font('DejaVu', 'B', 10)
    pdf.cell(0, 8, 'Trades (Scores, Gewichtungen, Analysen & Indikatoren)', ln=True)
    pdf.set_font('DejaVu', '', 7)

    if not trades_df.empty:
        cols = list(trades_df.columns)
        extra_cols = []
        if indicator_weights is not None:
            extra_cols.append("Gewichtungen")
        if "analysis_short" in trades_df.columns:
            extra_cols.append("Analyse (kurz)")
        if "analysis_long" in trades_df.columns:
            extra_cols.append("Analyse (lang)")
        all_cols = cols + extra_cols
        max_cols = 12
        special_col_widths = {
            "entry_date": 28, "exit_date": 28, "Date": 28, "Gewichtungen": 36
        }
        all_col_widths = [
            special_col_widths.get(col, (pdf.w - sum(special_col_widths.values())) / (max_cols - len(special_col_widths) + 1))
            for col in all_cols[:max_cols]
        ]
        for i, col in enumerate(all_cols[:max_cols]):
            pdf.cell(all_col_widths[i], 8, str(col), 1, 0, 'C')
        pdf.ln(8)
        pdf.set_font('DejaVu', '', 7)
        for idx, (_, row) in enumerate(trades_df.iterrows()):
            for i, col in enumerate(all_cols[:max_cols]):
                width = all_col_widths[i]
                if col in trades_df.columns:
                    val = row[col]
                    text = fmt_number(val)
                elif col == "Gewichtungen":
                    text = str(indicator_weights) if indicator_weights else "-"
                elif col == "Analyse (kurz)":
                    text = row.get("analysis_short", "-")
                elif col == "Analyse (lang)":
                    text = row.get("analysis_long", "-")
                else:
                    text = "-"
                if isinstance(text, str) and len(text) > 22:
                    text = text[:22] + "…"
                pdf.cell(width, 8, text, 1, 0, 'C')
            pdf.ln(8)
            if charts_dir:
                trade_id = row.get("trade_id") or row.get("Trade_ID") or row.name
                chart_paths = [
                    os.path.join(charts_dir, f"trade_{trade_id}", "candlestick_trade.png"),
                    os.path.join(charts_dir, f"trade_{trade_id}", "radar_raw.png"),
                    os.path.join(charts_dir, f"trade_{trade_id}", "radar_norm.png"),
                ]
                for chart in chart_paths:
                    if os.path.exists(chart):
                        pdf.ln(1)
                        pdf.image(chart, w=38)
            if (idx + 1) % 22 == 0:
                pdf.add_page()
                pdf.set_font('DejaVu', 'B', 7)
                for i, col in enumerate(all_cols[:max_cols]):
                    pdf.cell(all_col_widths[i], 8, str(col), 1, 0, 'C')
                pdf.ln(8)
                pdf.set_font('DejaVu', '', 7)
        pdf.ln(8)
    else:
        pdf.cell(0, 8, "Keine Trades vorhanden", ln=True)

    # Indikator-Legende ... (wie gehabt)
    # ...
    pdf.output(pdf_path)
    print(f"PDF erfolgreich erstellt: {pdf_path}")
