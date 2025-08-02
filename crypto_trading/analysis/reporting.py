"""
Modul: reporting.py

Erstellt professionelle Reports (PDF & CSV) für Krypto-Strategien.
- PDF mit Portfolio-/Asset-/Paramset-Kennzahlen, Tabellen, allen Visuals, Top/Flop-Trades, Monatsauswertung, Zeitstrahl etc.
- CSV-Export aller Trades, aller Kennzahlen, für Forschung/Excel/Statistik
- Erkennt FAST-Mode und passt Report dynamisch an

Funktionen:
- export_pdf_report(...)
- export_csv_reports(...)

Author: ChatGPT Research, 2025
"""

import pandas as pd
import numpy as np
from fpdf import FPDF
import os

def export_csv_reports(df_all_trades, df_perf, out_dir="./"):
    """
    Speichert alle Trades und alle Kennzahlen als CSV im Output-Dir.
    """
    trade_cols = ['asset','param_idx','trade_id','kaufpreis','verkaufspreis','gewinn_verlust',
                  'kurzanalyse','warum_gewinn_verlust','was_besser','score','entry_time','exit_time','richtung']
    perf_cols = df_perf.columns.tolist()
    df_all_trades[trade_cols].to_csv(os.path.join(out_dir, "trades_detailed.csv"), index=False)
    df_perf[perf_cols].to_csv(os.path.join(out_dir, "strategy_performance_grid.csv"), index=False)
    print("CSV-Exports gespeichert: trades_detailed.csv, strategy_performance_grid.csv")

def export_pdf_report(
    df_all_trades, df_perf, asset_param_results, tmpdir, 
    portfolio_imgs={}, kpi_dict={}, FAST_MODE=False, filename="strategy_report.pdf"
):
    """
    PDF-Bericht mit Portfolio-Tabellen, Asset/Paramset-Abschnitten, allen Visuals, KPIs, Top/Flop-Trades.
    portfolio_imgs: Dict mit Dateinamen für Portfolio-Plots (Equity, Monatsrenditen, Donut etc.)
    kpi_dict: Dict mit Portfolio-KPIs
    FAST_MODE: Flag für Quick-Report
    """
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=12)

    # === Deckblatt ===
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 12, "Crypto-Strategie-Report", ln=True, align='C')
    pdf.ln(8)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Datum: {pd.Timestamp.now():%Y-%m-%d %H:%M}", ln=True)
    pdf.cell(0, 10, f"Assets: {', '.join(sorted(set(df_all_trades['asset'])))}", ln=True)
    pdf.cell(0, 10, f"Paramsets: {df_perf.shape[0]}", ln=True)
    if FAST_MODE:
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(220,0,0)
        pdf.cell(0, 12, "FAST-ANALYSE (nur 1 Paramset/Asset, wenige Visuals)", ln=True)
        pdf.set_text_color(0,0,0)
    pdf.ln(5)

    # === Portfolio-Übersicht ===
    pdf.set_font("Arial", 'B', 13)
    pdf.cell(0, 9, "Portfolio-Performance:", ln=True)
    pdf.set_font("Arial", size=11)
    for key, val in kpi_dict.items():
        pdf.cell(55, 7, f"{key}: {val:,.2f}" if isinstance(val, (float,int)) else f"{key}: {val}", ln=False)
    pdf.ln(8)

    # === Portfolio-Visuals ===
    for label, img in portfolio_imgs.items():
        if img and os.path.isfile(img):
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0,8,label,ln=True)
            pdf.image(img, w=150)
            pdf.ln(2)

    # === Top/Flop-Trades gesamt ===
    def get_top_flop_trades(trades, n=5):
        if trades.empty: return pd.DataFrame(), pd.DataFrame()
        top = trades.nlargest(n, 'gewinn_verlust')
        flop = trades.nsmallest(n, 'gewinn_verlust')
        return top, flop

    top, flop = get_top_flop_trades(df_all_trades, 5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, f"Top 5 Trades (Portfolio):", ln=True)
    pdf.set_font("Arial", size=10)
    for _, t in top.iterrows():
        pdf.multi_cell(0, 5, f"ID {int(t['trade_id'])} {t['asset']}: {t['kaufpreis']} → {t['verkaufspreis']} | PnL {t['gewinn_verlust']:.2f}% | {t['kurzanalyse']}")
    pdf.ln(1)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, f"Flop 5 Trades (Portfolio):", ln=True)
    pdf.set_font("Arial", size=10)
    for _, t in flop.iterrows():
        pdf.multi_cell(0, 5, f"ID {int(t['trade_id'])} {t['asset']}: {t['kaufpreis']} → {t['verkaufspreis']} | PnL {t['gewinn_verlust']:.2f}% | {t['kurzanalyse']}")
    pdf.ln(4)

    # === Asset/Paramset-Abschnitte ===
    for asset, paramsets in asset_param_results.items():
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 9, f"Asset: {asset}", ln=True)
        for param_idx, summary in paramsets.items():
            trades, perf = summary['trades'], summary['perf']
            eqfile = summary.get('eqfile')
            monfile = summary.get('monfile')
            drfile = summary.get('drawdownfile')
            shfile = summary.get('sharpefile')
            histfile = summary.get('histfile')
            boxfile = summary.get('boxfile')
            heatmapfile = summary.get('heatmapfile')
            timelinefile = summary.get('timelinefile')
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 7, f"Paramset {param_idx} (Sharpe {perf.get('Sharpe',np.nan):.2f}, PnL {perf.get('Gesamt-PnL',np.nan):.2f}%)", ln=True)
            pdf.set_font("Arial", size=10)
            # Parameter
            paramnames = [k for k in perf.keys() if k not in ("Asset", "Anzahl Trades", "Gesamt-PnL", "Trefferquote", "Ø Trade-PnL", "Max. Gewinn", "Max. Verlust","Sharpe","MaxDrawdown","Letztes Kapital","Sortino","CAGR","Volatilität","MeanDrawdown")]
            pdf.multi_cell(0, 5, "Parameter: " + ", ".join([f"{k}={perf[k]}" for k in paramnames]))
            # Monatsauswertung-Bild
            if monfile and os.path.isfile(monfile):
                pdf.ln(1)
                pdf.image(monfile, w=90)
            # Equity Curve Bild
            if eqfile and os.path.isfile(eqfile):
                pdf.image(eqfile, w=110)
            # Drawdown, Rolling Sharpe etc.
            if drfile and os.path.isfile(drfile):
                pdf.image(drfile, w=90)
            if shfile and os.path.isfile(shfile):
                pdf.image(shfile, w=90)
            if histfile and os.path.isfile(histfile):
                pdf.image(histfile, w=70)
            if boxfile and os.path.isfile(boxfile):
                pdf.image(boxfile, w=60)
            if heatmapfile and os.path.isfile(heatmapfile):
                pdf.image(heatmapfile, w=90)
            if timelinefile and os.path.isfile(timelinefile):
                pdf.image(timelinefile, w=100)
            pdf.ln(1)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 7, "Kennzahlen:", ln=True)
            pdf.set_font("Arial", size=9)
            for k in ['Anzahl Trades', 'Gesamt-PnL', 'Trefferquote', 'Ø Trade-PnL', 'Sharpe', 'MaxDrawdown', 'Letztes Kapital', 'Sortino', 'CAGR', 'Volatilität','MeanDrawdown']:
                if k in perf:
                    pdf.cell(35, 7, f"{k}: {perf[k]:.4f}" if isinstance(perf[k],float) else f"{k}: {perf[k]}", ln=True)
            # Top-/Flop-Trades zu diesem Asset/Paramset
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 6, f"Top 3 Trades:", ln=True)
            pdf.set_font("Arial", size=8)
            topx, flopx = get_top_flop_trades(trades, 3)
            for _, t in topx.iterrows():
                pdf.multi_cell(0, 4, f"ID {int(t['trade_id'])}: {t['kaufpreis']} → {t['verkaufspreis']} | {t['gewinn_verlust']:.2f}% | {t['kurzanalyse']}")
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 6, f"Flop 3 Trades:", ln=True)
            pdf.set_font("Arial", size=8)
            for _, t in flopx.iterrows():
                pdf.multi_cell(0, 4, f"ID {int(t['trade_id'])}: {t['kaufpreis']} → {t['verkaufspreis']} | {t['gewinn_verlust']:.2f}% | {t['kurzanalyse']}")
            # Einzeltrades Tabelle (max. 10)
            pdf.set_font("Arial", 'B', 9)
            pdf.cell(0, 5, "Einzeltrades:", ln=True)
            pdf.set_font("Arial", size=7)
            for _, t in trades.head(10).iterrows():
                pdf.multi_cell(0, 4, f"ID {int(t['trade_id'])}: {t['kaufpreis']} → {t['verkaufspreis']} | {t['gewinn_verlust']:.2f}% | Score: {t['score']} | {t['kurzanalyse']}")
            pdf.ln(2)
    pdf.output(filename)
    print(f"PDF-Report gespeichert: {filename}")
