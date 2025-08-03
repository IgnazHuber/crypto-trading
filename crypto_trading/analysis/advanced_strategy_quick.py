import os
import pandas as pd
import numpy as np
from datetime import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import argparse
import matplotlib.pyplot as plt

def resolve_path(fn):
    """
    Sucht eine Datei robust im aktuellen Ordner, im typischen Data-Ordner,
    oder akzeptiert absolute Pfade. Gibt den gefundenen Pfad zurück,
    sonst FileNotFoundError mit getesteten Kandidaten.
    """
    # Absolute Pfade direkt prüfen
    if os.path.isabs(fn) and os.path.exists(fn):
        return fn
    # Direkt im aktuellen Arbeitsverzeichnis?
    if os.path.exists(fn):
        return fn
    # Kandidaten in typischen Datenpfaden
    candidates = [
        os.path.join(os.getcwd(), fn),
        os.path.join(os.getcwd(), "crypto_trading", "data", "raw", fn),
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", fn),
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", fn)
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Datei nicht gefunden: {fn}\nGetestet: {candidates}")

def plot_portfolio_equity(trades_df, filename):
    """Portfolio-Equity Verlauf (Dummy, anpassen für echtes Equity)"""
    eq = pd.Series([1.0], index=[pd.to_datetime(trades_df['entry_time'].iloc[0])])
    for idx, trade in trades_df.iterrows():
        pnl = trade['pnl_rel']
        exit_time = pd.to_datetime(trade['exit_time'])
        eq = pd.concat([eq, pd.Series(eq.iloc[-1] * (1 + pnl), index=[exit_time])])
    eq = eq.sort_index()
    plt.figure(figsize=(10, 3))
    plt.plot(eq.index, eq.values, label="Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Equity")
    plt.title("Portfolio Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def export_pdf_report(all_trades, performance_grid, asset_param_results, outdir, filename="strategy_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", 'B', 18)
    pdf.cell(0, 12, "Crypto-Strategie-Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 10, f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, f"Assets: {', '.join(set(all_trades['asset']))}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, f"Parameter-Grid: {len(asset_param_results)} Kombinationen", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", 'B', 13)
    pdf.cell(0, 9, "Portfolio-Performance:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", size=11)
    gewinn_abs = all_trades['pnl_abs'].sum()
    gewinn_rel = 100 * all_trades['pnl_rel'].sum()
    pdf.cell(60, 7, f"PnL: {gewinn_abs:,.2f}  ({gewinn_rel:.2f}%)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    # Hier könnten weitere KPIs ergänzt werden ...
    pdf.set_font("helvetica", size=10)
    pdf.cell(0, 10, "Weitere Details siehe CSV/Plots.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    # Plots einbinden (hier nur als Beispiel: Equity Curve)
    port_eqfile = os.path.join(outdir, "portfolio_equity.png")
    plot_portfolio_equity(all_trades, port_eqfile)
    pdf.image(port_eqfile, w=180)
    # Save PDF
    out_pdf = os.path.join(outdir, filename)
    pdf.output(out_pdf)
    print(f"PDF-Report gespeichert: {out_pdf}")

def analyze_trades(df, quick=False):
    """Simulierter Analyseprozess für alle Trades (Dummy-Logik)"""
    # Hier könntest du eigentliche Strategie/Backtest etc. aufrufen
    # Wir fügen ein paar Dummy-Spalten hinzu, damit das Reporting funktioniert
    df['asset'] = df.get('asset', "BTCUSDT")
    df['pnl_abs'] = np.random.normal(0, 100, len(df))
    df['pnl_rel'] = np.random.normal(0, 0.02, len(df))
    # Zeitspalten als Beispiel
    if 'entry_time' not in df or 'exit_time' not in df:
        start = pd.to_datetime("2024-01-01")
        df['entry_time'] = [start + pd.Timedelta(hours=i) for i in range(len(df))]
        df['exit_time'] = [start + pd.Timedelta(hours=i+1) for i in range(len(df))]
    return df

def main():
    parser = argparse.ArgumentParser(description="Schneller Strategie-Backtest mit robustem Pfadhandling.")
    parser.add_argument("--quick", action="store_true", help="Quick-Run mit reduziertem Datensatz für schnellen Test (<1min)")
    parser.add_argument("--input", type=str, nargs='+', help="Dateien (Parquet) zur Analyse",
                        default=["BTCUSDT_1h_1year_ccxt.parquet"])
    parser.add_argument("--outdir", type=str, default="results", help="Output-Verzeichnis für Berichte und Plots")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if args.quick:
        print("\n*** Quickmode: Die Analyse läuft in <1min und nur mit Beispieldaten! ***\n")

    all_trades = []
    for fn in args.input:
        print(f"Lade und analysiere: {fn}")
        try:
            resolved = resolve_path(fn)
            df = pd.read_parquet(resolved)
        except Exception as e:
            print(f"WARNUNG: Konnte Datei {fn} nicht laden: {e}")
            continue
        if args.quick:
            df = df.head(100)
        trades = analyze_trades(df, quick=args.quick)
        all_trades.append(trades)

    if not all_trades:
        print("Keine Daten geladen – Skript beendet.")
        return

    all_trades_df = pd.concat(all_trades, ignore_index=True)
    # Dummy-Grid/Results
    performance_grid = pd.DataFrame({
        "param_set": ["A", "B", "C"],
        "PnL": [all_trades_df['pnl_abs'].sum()] * 3
    })
    asset_param_results = [{"asset": x, "params": {}} for x in set(all_trades_df['asset'])]

    # CSV & Excel Export
    trades_csv = os.path.join(args.outdir, "trades_detailed.csv")
    grid_csv = os.path.join(args.outdir, "strategy_performance_grid.csv")
    grid_xlsx = os.path.join(args.outdir, "strategy_performance_grid.xlsx")

    all_trades_df.to_csv(trades_csv, index=False)
    performance_grid.to_csv(grid_csv, index=False)
    performance_grid.to_excel(grid_xlsx, index=False)  # <- NEU
    print(f"Trade-Details gespeichert: {trades_csv}")
    print(f"Performance-Grid gespeichert: {grid_csv} und {grid_xlsx}")


    # PDF-Report
    export_pdf_report(all_trades_df, performance_grid, asset_param_results, args.outdir, filename="strategy_report.pdf")

if __name__ == "__main__":
    main()
