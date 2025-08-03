# crypto_trading/analysis/reporting.py

import os
import pandas as pd
import numpy as np
from fpdf import FPDF

def export_csv_reports(df_all_trades, df_perf, out_dir="./"):
    os.makedirs(out_dir, exist_ok=True)
    df_all_trades.to_csv(os.path.join(out_dir, "trades_detailed.csv"), index=False)
    df_perf.to_csv(os.path.join(out_dir, "strategy_performance_grid.csv"), index=False)
    print("CSV-Exports gespeichert:", out_dir)

def export_excel(trades, perf, out_dir="./report_output"):
    os.makedirs(out_dir, exist_ok=True)
    excel_path = os.path.join(out_dir, "full_results.xlsx")
    beschreibung = [
        "Asset Name", "Parameter-Index", "Trade-Nummer", "Kaufpreis", "Verkaufspreis",
        "Absoluter/prozentualer Gewinn/Verlust", "Kurzanalyse", "Begründung", "Verbesserungsvorschlag",
        "Score", "Kaufzeit", "Verkaufszeit", "Richtung (Long/Short/Flat)"
    ]
    bedeutung_trades = beschreibung + [""] * (len(trades.columns) - len(beschreibung))
    doku_trades = pd.DataFrame({
        "Spalte": list(trades.columns),
        "Beispiel": [str(trades.iloc[0][c]) if len(trades)>0 else "" for c in trades.columns],
        "Bedeutung": bedeutung_trades[:len(trades.columns)]
    })
    doku_perf = pd.DataFrame({
        "Spalte": list(perf.columns),
        "Bedeutung": list(perf.columns)
    })
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        trades.to_excel(writer, sheet_name="Trades", index=False)
        perf.to_excel(writer, sheet_name="Performance", index=False)
        doku_trades.to_excel(writer, sheet_name="Doku_Trades", index=False)
        doku_perf.to_excel(writer, sheet_name="Doku_Performance", index=False)
    print(f"Excel-Export gespeichert: {excel_path}")

def safe_ascii(text, max_word_len=50):
    """Macht Text FPDF-/ASCII-sicher, kürzt zu lange Wörter, entfernt Zeilenumbrüche und nicht-latin-1."""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("→", "->")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("…", "...")
    text = text.replace("✓", "v")
    text = text.replace("„", '"').replace("“", '"')
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("•", "*")
    # Wortweises Kürzen
    words = []
    for w in text.split():
        if len(w) > max_word_len:
            w = w[:max_word_len] + "…"
        words.append(w)
    text = " ".join(words)
    return text.encode("latin-1", errors="replace").decode("latin-1")

def get_top_flop_trades(trades, n=5):
    if trades.empty or 'gewinn_verlust' not in trades.columns:
        return pd.DataFrame(), pd.DataFrame()
    trades = trades.copy()
    trades['gewinn_verlust'] = pd.to_numeric(trades['gewinn_verlust'], errors="coerce")
    if trades['gewinn_verlust'].dropna().empty:
        return pd.DataFrame(), pd.DataFrame()
    top = trades.nlargest(n, 'gewinn_verlust')
    flop = trades.nsmallest(n, 'gewinn_verlust')
    return top, flop

def pdf_multicell_safe(pdf, txt):
    txt = safe_ascii(txt)
    if not txt.strip():
        txt = "."
    try:
        pdf.multi_cell(0, 5, txt)
    except Exception as e:
        print(f"[PDF WARN] {e} bei Text: {repr(txt)}")
        try:
            pdf.multi_cell(0, 5, ".")
        except Exception as e2:
            print(f"[PDF ERROR] {e2} bei endgültigem Fallback – PDF weiter ohne Zeile")


def export_pdf_report(
    df_all_trades, df_perf, asset_param_results, tmpdir,
    portfolio_imgs={}, kpi_dict={}, FAST_MODE=False, filename="strategy_report.pdf"
):
    if FAST_MODE:
        print("⚡ FAST-Mode: PDF-Export übersprungen.")
        return None

    out_dir = os.path.dirname(filename) or "."
    os.makedirs(out_dir, exist_ok=True)
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 12, "Crypto-Strategie-Report", ln=True, align='C')
    pdf.ln(8)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, safe_ascii(f"Datum: {pd.Timestamp.now():%Y-%m-%d %H:%M}"), ln=True)
    if 'asset' in df_all_trades.columns:
        assetlist = ', '.join(sorted(map(str, set(df_all_trades['asset']))))
    else:
        assetlist = 'n/a'
    pdf.cell(0, 10, safe_ascii(f"Assets: {assetlist}"), ln=True)
    pdf.cell(0, 10, safe_ascii(f"Paramsets: {df_perf.shape[0]}"), ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 13)
    pdf.cell(0, 9, safe_ascii("Portfolio-KPIs:"), ln=True)
    pdf.set_font("Arial", size=11)
    for key, val in kpi_dict.items():
        pdf.cell(55, 7, safe_ascii(f"{key}: {val:,.2f}" if isinstance(val, (float,int)) else f"{key}: {val}"), ln=False)
    pdf.ln(8)
    for label, img in portfolio_imgs.items():
        if img and os.path.isfile(img):
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0,8,safe_ascii(label),ln=True)
            pdf.image(img, w=150)
            pdf.ln(2)
    # Top/Flop/Einzeltrades robust
    top, flop = get_top_flop_trades(df_all_trades, 5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, safe_ascii("Top 5 Trades (Portfolio):"), ln=True)
    pdf.set_font("Arial", size=10)
    for _, t in top.iterrows():
        txt = f"ID {int(t.get('trade_id', 0)) if 'trade_id' in t else '?'} {t.get('asset','')}: {t.get('kaufpreis','?')} -> {t.get('verkaufspreis','?')} | PnL {t.get('gewinn_verlust',np.nan):.2f}% | {t.get('kurzanalyse','')}"
        pdf_multicell_safe(pdf, txt)
    pdf.ln(1)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, safe_ascii("Flop 5 Trades (Portfolio):"), ln=True)
    pdf.set_font("Arial", size=10)
    for _, t in flop.iterrows():
        txt = f"ID {int(t.get('trade_id', 0)) if 'trade_id' in t else '?'} {t.get('asset','')}: {t.get('kaufpreis','?')} -> {t.get('verkaufspreis','?')} | PnL {t.get('gewinn_verlust',np.nan):.2f}% | {t.get('kurzanalyse','')}"
        pdf_multicell_safe(pdf, txt)
    pdf.ln(4)
    # ... weitere Abschnitte, Tabellen, Grafiken etc. können hier weiter ergänzt werden ...
    pdf.output(filename)
    print(f"PDF-Report gespeichert: {filename}")
