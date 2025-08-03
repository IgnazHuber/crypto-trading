import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import os
from itertools import product
from datetime import datetime
import tempfile
import shutil
from tqdm import tqdm
from fpdf import FPDF

# ======== Einstellungen ========
DATA_PATH = r"c:\Projekte\crypto_trading\crypto_trading\data\raw"
DEFAULT_FILES = ["BTCUSDT_1h_1year_ccxt.parquet"]  # beliebig erweitern
EXPORT_TRADE_CSV = True
EXPORT_PERF_CSV = True
EXPORT_PDF = True
SCORE_ENTRY_THRESHOLD = 3
N_TOP = 5   # für Top-/Flop-Trades

# === Parameter-Grid ===
GRID = {
    "RSI_PERIOD": [8, 14, 21],
    "MACD_FAST": [8, 12],
    "MACD_SLOW": [18, 26],
    "MACD_SIGNAL": [6, 9],
    "EMA_SHORT": [21, 50],
    "EMA_LONG": [100, 200],
    "BB_WINDOW": [10, 20],
    "STOCH_WINDOW": [8, 14],
    "STOP_LOSS_PCT": [0.02, 0.03, 0.05],
    "TAKE_PROFIT_PCT": [0.04, 0.06, 0.1],
}
param_grid = list(product(*GRID.values()))
param_names = list(GRID.keys())

# ======= Datei-Auswahl =======
files = [f for f in os.listdir(DATA_PATH) if f.endswith('.parquet')]
print("Verfügbare Parquet-Dateien:")
for idx, fname in enumerate(files):
    print(f"{idx+1:2d}. {fname}")
multi_choice = input(f"Mehrere Dateien im Batch? (z.B. 1,2,5 oder Enter für Defaults ({', '.join(DEFAULT_FILES)})): ")
if multi_choice.strip():
    selection = [int(i.strip())-1 for i in multi_choice.split(',') if i.strip().isdigit()]
    parquet_files = [os.path.join(DATA_PATH, files[i]) for i in selection if 0 <= i < len(files)]
else:
    parquet_files = [os.path.join(DATA_PATH, f) for f in DEFAULT_FILES]

# ===== Ergebnis-Container =====
all_results = []
all_trades = []
asset_param_results = {}

# ===== Kernfunktionen =====

def run_score_strategy(df, params, asset, threshold=SCORE_ENTRY_THRESHOLD):
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, EMA_SHORT, EMA_LONG, BB_WINDOW, STOCH_WINDOW, STOP_LOSS_PCT, TAKE_PROFIT_PCT = params
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=RSI_PERIOD).rsi()
    macd = ta.trend.MACD(df['close'], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema_short'] = ta.trend.EMAIndicator(df['close'], window=EMA_SHORT).ema_indicator()
    df['ema_long'] = ta.trend.EMAIndicator(df['close'], window=EMA_LONG).ema_indicator()
    bb = ta.volatility.BollingerBands(df['close'], window=BB_WINDOW)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=STOCH_WINDOW)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df = df.dropna()

    entries = []
    position = None
    entry_price = 0
    entry_score = 0
    entry_reasons = ""
    for i, row in df.iterrows():
        score = 0
        reasons = []
        if row['rsi'] < 35:
            score += 1
            reasons.append("RSI < 35")
        if row['macd'] > row['macd_signal']:
            score += 1
            reasons.append("MACD Bull-Crossover")
        if row['ema_short'] > row['ema_long']:
            score += 1
            reasons.append(f"EMA({EMA_SHORT}) > EMA({EMA_LONG})")
        if row['close'] <= row['bb_lower']:
            score += 1
            reasons.append("Kurs <= BB lower")
        if row['stoch_k'] < 25 and row['stoch_k'] > row['stoch_d']:
            score += 1
            reasons.append("Stoch %K < 25 & > %D")
        entry = (score >= threshold)
        exit_ = (
            (row['rsi'] > 60) or
            (row['macd'] < row['macd_signal']) or
            (row['close'] >= row['bb_upper']) or
            (row['stoch_k'] > 80) or
            (row['close'] < entry_price * (1 - STOP_LOSS_PCT)) or
            (row['close'] > entry_price * (1 + TAKE_PROFIT_PCT))
        )
        if position and exit_:
            pnl_pct = (row['close'] - entry_price) / entry_price * 100
            reason = []
            if row['rsi'] > 60: reason.append("RSI > 60")
            if row['macd'] < row['macd_signal']: reason.append("MACD Bear-Crossover")
            if row['close'] >= row['bb_upper']: reason.append("Kurs >= BB upper")
            if row['stoch_k'] > 80: reason.append("Stoch > 80")
            if row['close'] < entry_price * (1 - STOP_LOSS_PCT): reason.append("Stop-Loss")
            if row['close'] > entry_price * (1 + TAKE_PROFIT_PCT): reason.append("Take-Profit")
            warum = "\n".join(reason)
            was_besser = "TP/SL optimieren" if abs(pnl_pct) < 0.5 else "Timing verbessern"
            entries[-1].update({
                'exit_time': i,
                'exit_price': row['close'],
                'gewinn_verlust': pnl_pct,
                'kurzanalyse': entry_reasons,
                'warum_gewinn_verlust': warum,
                'was_besser': was_besser,
                'score': entry_score
            })
            position = None
        if not position and entry:
            entry_score = score
            entry_reasons = "\n".join(reasons)
            entries.append({
                'asset': asset,
                'entry_time': i,
                'entry_price': row['close'],
                'score': entry_score,
                'kurzanalyse': entry_reasons
            })
            entry_price = row['close']
            position = True
    if position and entries:
        last = df.iloc[-1]
        pnl_pct = (last['close'] - entry_price) / entry_price * 100
        entries[-1].update({
            'exit_time': last.name,
            'exit_price': last['close'],
            'gewinn_verlust': pnl_pct,
            'kurzanalyse': entry_reasons,
            'warum_gewinn_verlust': "Ende Zeitraum",
            'was_besser': "Früher exitten",
            'score': entry_score
        })
    trades = pd.DataFrame(entries)
    trades.insert(0, 'trade_id', range(1, len(trades)+1))
    trades['kaufpreis'] = trades['entry_price']
    trades['verkaufspreis'] = trades['exit_price']
    return trades

def equity_curve(trades):
    if trades.empty: return pd.Series(dtype=float)
    eq = pd.Series(1.0, index=pd.to_datetime(trades['exit_time']))
    for idx, trade in trades.iterrows():
        pnl = trade['gewinn_verlust'] / 100.0
        eq.iloc[idx:] = eq.iloc[idx:] * (1 + pnl)
    return eq

def performance_metrics(trades, params, asset):
    if trades.empty:
        return {
            "Asset": asset,
            **{k: v for k, v in zip(param_names, params)},
            "Anzahl Trades": 0, "Gesamt-PnL": np.nan, "Trefferquote": np.nan,
            "Ø Trade-PnL": np.nan, "Max. Gewinn": np.nan, "Max. Verlust": np.nan,
            "Sharpe": np.nan, "MaxDrawdown": np.nan, "Letztes Kapital": np.nan
        }
    trades = trades.copy()
    trades['holding_period'] = (pd.to_datetime(trades['exit_time']) - pd.to_datetime(trades['entry_time'])).dt.total_seconds() / 3600
    total_pnl = trades['gewinn_verlust'].sum()
    num_trades = len(trades)
    win_rate = (trades['gewinn_verlust'] > 0).mean() * 100
    avg_pnl = trades['gewinn_verlust'].mean()
    max_gain = trades['gewinn_verlust'].max()
    max_loss = trades['gewinn_verlust'].min()
    eq = equity_curve(trades)
    returns = eq.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(24*365) if returns.std() > 0 else np.nan
    cummax = eq.cummax()
    drawdown = (eq - cummax) / cummax
    max_drawdown = drawdown.min()
    last_equity = eq.dropna().iloc[-1] if not eq.dropna().empty else np.nan
    return {
        "Asset": asset,
        **{k: v for k, v in zip(param_names, params)},
        "Anzahl Trades": num_trades, "Gesamt-PnL": total_pnl, "Trefferquote": win_rate,
        "Ø Trade-PnL": avg_pnl, "Max. Gewinn": max_gain, "Max. Verlust": max_loss,
        "Sharpe": sharpe, "MaxDrawdown": max_drawdown, "Letztes Kapital": last_equity
    }

def plot_equity_curve(eq, filename, title="Equity Curve"):
    fig, ax = plt.subplots(figsize=(10, 3))
    eq.plot(ax=ax, title=title)
    ax.set_ylabel("Kapital")
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def plot_monthly(trades, filename, title="Monatliche Strategie-Rendite"):
    if trades.empty: return
    trades['monat'] = pd.to_datetime(trades['entry_time']).dt.to_period('M')
    monat = trades.groupby('monat')['gewinn_verlust'].sum()
    fig, ax = plt.subplots(figsize=(10,3))
    monat.plot(kind='bar', ax=ax, title=title)
    ax.set_ylabel("PnL [%]")
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def gesamtperformance(trades):
    einsatz = trades['kaufpreis'].sum()
    erloese = trades['verkaufspreis'].sum()
    gewinn_abs = erloese - einsatz
    gewinn_rel = (gewinn_abs / einsatz) * 100 if einsatz else 0
    return einsatz, erloese, gewinn_abs, gewinn_rel

def get_top_flop_trades(trades, n=5):
    if trades.empty: return pd.DataFrame(), pd.DataFrame()
    top = trades.nlargest(n, 'gewinn_verlust')
    flop = trades.nsmallest(n, 'gewinn_verlust')
    return top, flop

def plot_portfolio_equity(df_all_trades, filename):
    # Kombinierte Portfolio-Kurve aus allen Assets und Paramsets
    eq = pd.Series(1.0)
    all_trades = df_all_trades.copy()
    all_trades = all_trades.sort_values('exit_time')
    for idx, trade in all_trades.iterrows():
        pnl = trade['gewinn_verlust'] / 100.0
        eq = eq.append(pd.Series(eq.iloc[-1] * (1 + pnl), index=[pd.to_datetime(trade['exit_time'])]))
    eq = eq[~eq.index.duplicated()]
    fig, ax = plt.subplots(figsize=(13,4))
    eq.plot(ax=ax, title="Portfolio Equity Curve (Alle Trades/Assets)")
    ax.set_ylabel("Kapital")
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def plot_portfolio_monthly(df_all_trades, filename):
    all_trades = df_all_trades.copy()
    all_trades['monat'] = pd.to_datetime(all_trades['entry_time']).dt.to_period('M')
    monat = all_trades.groupby('monat')['gewinn_verlust'].sum()
    fig, ax = plt.subplots(figsize=(12,4))
    monat.plot(kind='bar', ax=ax, title="Portfolio Monatsrenditen (Alle Trades/Assets)")
    ax.set_ylabel("PnL [%]")
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def export_pdf_report(all_trades, all_perf, asset_param_results, tmpdir, filename="strategy_report.pdf"):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 12, "Crypto-Strategie-Report", ln=True, align='C')
    pdf.ln(8)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 10, f"Assets: {', '.join(set(all_trades['asset']))}", ln=True)
    pdf.cell(0, 10, f"Parameter-Grid: {len(param_grid)} Kombinationen", ln=True)
    pdf.ln(4)
    # Portfolio-Gesamtauswertung
    einsatz, erloese, gewinn_abs, gewinn_rel = gesamtperformance(all_trades)
    pdf.set_font("Arial", 'B', 13)
    pdf.cell(0, 9, "Portfolio-Performance:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(60, 7, f"Einsatz: {einsatz:,.2f}")
    pdf.cell(60, 7, f"Verkauf: {erloese:,.2f}")
    pdf.cell(60, 7, f"PnL: {gewinn_abs:,.2f}  ({gewinn_rel:.2f}%)", ln=True)
    pdf.ln(3)
    # Portfolio-Grafiken
    port_eqfile = os.path.join(tmpdir, "portfolio_equity.png")
    port_monfile = os.path.join(tmpdir, "portfolio_monat.png")
    plot_portfolio_equity(all_trades, port_eqfile)
    plot_portfolio_monthly(all_trades, port_monfile)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Portfolio Equity Curve:", ln=True)
    pdf.image(port_eqfile, w=150)
    pdf.cell(0, 8, "Portfolio Monatsrenditen:", ln=True)
    pdf.image(port_monfile, w=150)
    pdf.ln(4)
    # Top-/Flop-Trades gesamt
    top, flop = get_top_flop_trades(all_trades, N_TOP)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, f"Top {N_TOP} Trades (Portfolio):", ln=True)
    pdf.set_font("Arial", size=10)
    for _, t in top.iterrows():
        pdf.multi_cell(0, 5, f"ID {int(t['trade_id'])} {t['asset']}: {t['kaufpreis']} → {t['verkaufspreis']} | PnL {t['gewinn_verlust']:.2f}% | {t['kurzanalyse']}")
    pdf.ln(2)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, f"Flop {N_TOP} Trades (Portfolio):", ln=True)
    pdf.set_font("Arial", size=10)
    for _, t in flop.iterrows():
        pdf.multi_cell(0, 5, f"ID {int(t['trade_id'])} {t['asset']}: {t['kaufpreis']} → {t['verkaufspreis']} | PnL {t['gewinn_verlust']:.2f}% | {t['kurzanalyse']}")
    pdf.ln(6)
    # Asset-seitig & Paramset
    for asset, paramsets in asset_param_results.items():
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 9, f"Asset: {asset}", ln=True)
        for param_idx, summary in paramsets.items():
            trades, perf, eqfile, monfile = summary['trades'], summary['perf'], summary['eqfile'], summary['monfile']
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 7, f"Paramset {param_idx} (Sharpe {perf['Sharpe']:.2f}, PnL {perf['Gesamt-PnL']:.2f}%)", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 5, "Parameter: " + ", ".join([f"{k}={perf[k]}" for k in param_names]))
            pdf.ln(2)
            pdf.image(monfile, w=90)
            pdf.ln(1)
            pdf.image(eqfile, w=110)
            pdf.ln(1)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 7, "Kennzahlen:", ln=True)
            pdf.set_font("Arial", size=9)
            for k in ['Anzahl Trades', 'Gesamt-PnL', 'Trefferquote', 'Ø Trade-PnL', 'Sharpe', 'MaxDrawdown', 'Letztes Kapital']:
                pdf.cell(35, 7, f"{k}: {perf[k]}", ln=True)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 6, f"Top {N_TOP} Trades:", ln=True)
            pdf.set_font("Arial", size=8)
            topx, flopx = get_top_flop_trades(trades, N_TOP)
            for _, t in topx.iterrows():
                pdf.multi_cell(0, 4, f"ID {int(t['trade_id'])}: {t['kaufpreis']} → {t['verkaufspreis']} | {t['gewinn_verlust']:.2f}% | {t['kurzanalyse']}")
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 6, f"Flop {N_TOP} Trades:", ln=True)
            pdf.set_font("Arial", size=8)
            for _, t in flopx.iterrows():
                pdf.multi_cell(0, 4, f"ID {int(t['trade_id'])}: {t['kaufpreis']} → {t['verkaufspreis']} | {t['gewinn_verlust']:.2f}% | {t['kurzanalyse']}")
            pdf.set_font("Arial", 'B', 9)
            pdf.cell(0, 5, "Einzeltrades:", ln=True)
            pdf.set_font("Arial", size=7)
            for _, t in trades.head(10).iterrows():
                pdf.multi_cell(0, 4, f"ID {int(t['trade_id'])}: {t['kaufpreis']} → {t['verkaufspreis']} | {t['gewinn_verlust']:.2f}% | Score: {t['score']} | {t['kurzanalyse']}")
            pdf.ln(2)
    pdf.output(filename)
    print(f"\nPDF-Report gespeichert: {filename}")

# ========== BATCH + GRID + PROGRESS ==========
tmpdir = tempfile.mkdtemp()
total_tasks = len(parquet_files) * len(param_grid)
progress = tqdm(total=total_tasks, desc="Analyse & Reporting", ncols=80)

for parquet_file in parquet_files:
    asset = os.path.basename(parquet_file).split("_")[0]
    df = pd.read_parquet(parquet_file)
    df = df.sort_index()
    if not np.issubdtype(df.index.dtype, np.datetime64):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            df.set_index('timestamp', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=['close', 'high', 'low'])
    param_results = {}
    for param_idx, params in enumerate(param_grid):
        trades = run_score_strategy(df, params, asset)
        if not trades.empty:
            trades['asset'] = asset
            trades['param_idx'] = param_idx
            trades['param_str'] = str(params)
        all_trades.append(trades)
        perf = performance_metrics(trades, params, asset)
        eq = equity_curve(trades)
        eqfile = os.path.join(tmpdir, f"{asset}_{param_idx}_eq.png")
        monfile = os.path.join(tmpdir, f"{asset}_{param_idx}_mon.png")
        plot_equity_curve(eq, eqfile, title=f"{asset} Equity Curve (Paramset {param_idx})")
        plot_monthly(trades, monfile, title=f"{asset} Monatsrenditen (Paramset {param_idx})")
        param_results[param_idx] = {'trades': trades, 'perf': perf, 'eqfile': eqfile, 'monfile': monfile}
        all_results.append(perf)
        progress.update(1)
    asset_param_results[asset] = param_results

progress.close()

if EXPORT_TRADE_CSV:
    df_all_trades = pd.concat(all_trades, ignore_index=True)
    trade_cols = ['asset','param_idx','trade_id','kaufpreis','verkaufspreis','gewinn_verlust',
                  'kurzanalyse','warum_gewinn_verlust','was_besser','score','entry_time','exit_time']
    df_all_trades[trade_cols].to_csv("trades_detailed.csv", index=False)
    print("\nTrade-Details gespeichert: trades_detailed.csv")

if EXPORT_PERF_CSV:
    df_perf = pd.DataFrame(all_results)
    df_perf.to_csv("strategy_performance_grid.csv", index=False)
    print("Performance-Grid gespeichert: strategy_performance_grid.csv")

if EXPORT_PDF:
    df_all_trades = pd.concat(all_trades, ignore_index=True)
    export_pdf_report(df_all_trades, df_perf, asset_param_results, tmpdir, filename="strategy_report.pdf")
    shutil.rmtree(tmpdir)

print("\nFertig. Alle Reports sind geschrieben.")
