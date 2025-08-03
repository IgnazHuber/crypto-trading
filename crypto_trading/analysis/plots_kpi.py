# crypto_trading/analysis/plots_kpi.py

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def _is_numerical(series):
    """Testet, ob Serie (nach coercen) mindestens einen numerischen Wert enthält."""
    arr = pd.to_numeric(series, errors="coerce").dropna()
    return arr if arr.size > 0 else None

def plot_equity_curve(eq, filename, title="Equity Curve"):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if not isinstance(eq, (pd.Series, np.ndarray)) or len(eq) < 2 or np.all(np.isnan(eq)):
        print(f"[plot_equity_curve] Keine Daten – Plot {filename} übersprungen.")
        return None
    plt.figure(figsize=(10, 3))
    pd.Series(eq).plot(title=title)
    plt.ylabel("Kapital")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_drawdown(eq, filename, title="Drawdown-Kurve"):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if not isinstance(eq, (pd.Series, np.ndarray)) or len(eq) < 2 or np.all(np.isnan(eq)):
        print(f"[plot_drawdown] Keine Daten – Plot {filename} übersprungen.")
        return None
    dd = (eq / np.maximum.accumulate(eq)) - 1
    if np.all(np.isnan(dd)) or len(dd) < 2:
        print(f"[plot_drawdown] Keine Daten – Plot {filename} übersprungen.")
        return None
    plt.figure(figsize=(10,3))
    plt.fill_between(range(len(dd)), dd, 0, color="red", alpha=0.4)
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_pnl_hist(trades, filename, title="PnL-Histogramm"):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if trades.empty or trades.shape[1] == 0:
        print(f"[plot_pnl_hist] Keine Spalten – Plot {filename} übersprungen.")
        return None
    col = 'gewinn_verlust' if 'gewinn_verlust' in trades.columns else trades.columns[-1]
    series = _is_numerical(trades[col])
    if series is None:
        print(f"[plot_pnl_hist] Keine numerischen Werte – Plot {filename} übersprungen.")
        return None
    plt.figure(figsize=(7,3))
    plt.hist(series, bins=30, alpha=0.7)
    plt.title(title)
    plt.xlabel("Gewinn/Verlust [%]")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_month_heatmap(trades, filename, title="Monats-Performance Heatmap"):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if trades.empty or trades.shape[1] == 0:
        print(f"[plot_month_heatmap] Keine Trades – Heatmap {filename} übersprungen.")
        return None
    date_col = None
    for col in trades.columns:
        if "entry_time" in col or "date" in col:
            date_col = col
            break
    if date_col is None:
        print(f"[plot_month_heatmap] Keine Zeitspalte gefunden – Heatmap {filename} übersprungen.")
        return None
    df = trades.copy()
    try:
        df['year'] = pd.to_datetime(df[date_col]).dt.year
        df['month'] = pd.to_datetime(df[date_col]).dt.month
    except Exception as e:
        print(f"[plot_month_heatmap] Fehler beim Datetime-Parsing: {e}")
        return None
    col = 'gewinn_verlust' if 'gewinn_verlust' in df.columns else df.columns[-1]
    series = _is_numerical(df[col])
    if series is None:
        print(f"[plot_month_heatmap] Keine numerischen Werte – Heatmap {filename} übersprungen.")
        return None
    heat = df.groupby(['year','month'])[col].sum().unstack().fillna(0)
    if heat.empty:
        print(f"[plot_month_heatmap] Keine Werte für Monats-Heatmap – {filename}")
        return None
    plt.figure(figsize=(10,4))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
    plt.title(title)
    plt.xlabel("Monat")
    plt.ylabel("Jahr")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_sharpe_pnl_scatter(perf_df, filename, title="Top-Paramsets: Sharpe vs. PnL"):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if perf_df.empty or perf_df.shape[1] == 0 or not {'Sharpe', 'Gesamt-PnL'}.issubset(perf_df.columns):
        print(f"[plot_sharpe_pnl_scatter] Felder fehlen – Plot {filename} übersprungen.")
        return None
    x = _is_numerical(perf_df['Sharpe'])
    y = _is_numerical(perf_df['Gesamt-PnL'])
    if x is None or y is None or len(x) != len(y):
        print(f"[plot_sharpe_pnl_scatter] Keine validen Werte – Plot {filename} übersprungen.")
        return None
    plt.figure(figsize=(7,4))
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Gesamt-PnL [%]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_portfolio_donut(trades, filename, title="Portfolio-Allokation nach Asset (PnL)"):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if trades.empty or trades.shape[1] == 0 or 'asset' not in trades.columns or 'gewinn_verlust' not in trades.columns:
        print(f"[plot_portfolio_donut] Felder fehlen – Plot {filename} übersprungen.")
        return None
    trades = trades.copy()
    trades['gewinn_verlust'] = pd.to_numeric(trades['gewinn_verlust'], errors='coerce')
    # Nur positive Werte!
    sizes = trades.groupby('asset')['gewinn_verlust'].sum()
    sizes = sizes[sizes > 0]
    if sizes.empty or not np.issubdtype(sizes.dtype, np.number):
        print(f"[plot_portfolio_donut] Keine positiven Werte für Donut – Plot {filename} übersprungen.")
        return None
    plt.figure(figsize=(5,5))
    plt.pie(sizes, labels=sizes.index, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename


def plot_pnl_box(trades, filename, title="Outlier/Boxplot aller Gewinne/Verluste"):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if trades.empty or trades.shape[1] == 0:
        print(f"[plot_pnl_box] Keine Spalten – kein Boxplot gespeichert: {filename}")
        return None
    col = 'gewinn_verlust' if 'gewinn_verlust' in trades.columns else trades.columns[-1]
    series = _is_numerical(trades[col])
    if series is None:
        print(f"[plot_pnl_box] Keine numerischen Werte – kein Boxplot gespeichert: {filename}")
        return None
    plt.figure(figsize=(7,3))
    plt.boxplot(series, vert=False)
    plt.title(title)
    plt.xlabel("Gewinn/Verlust [%]")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_corr_matrix(perf_df, filename, metric="Gesamt-PnL", title="Korrelation zwischen Assets"):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if perf_df.empty or perf_df.shape[1] == 0 or 'Asset' not in perf_df.columns or perf_df['Asset'].nunique() < 2 or metric not in perf_df.columns:
        print(f"[plot_corr_matrix] Zu wenig Assets oder Feld fehlt – Plot {filename} übersprungen.")
        return None
    piv = perf_df.pivot(index='param_idx', columns='Asset', values=metric)
    corr = piv.corr()
    if corr.empty:
        print(f"[plot_corr_matrix] Keine Werte – Plot {filename} übersprungen.")
        return None
    plt.figure(figsize=(7,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_tradetimeline(trades, filename, title="Zeitstrahl: Wann wurde investiert?"):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if trades.empty or trades.shape[1] == 0:
        print(f"[plot_tradetimeline] Keine Trades – Plot {filename} übersprungen.")
        return None
    date_col = None
    for col in trades.columns:
        if "entry_time" in col or "date" in col:
            date_col = col
            break
    if date_col is None:
        print(f"[plot_tradetimeline] Keine Zeitspalte gefunden – Plot {filename} übersprungen.")
        return None
    series = pd.to_datetime(trades[date_col], errors='coerce').dropna()
    if series.empty:
        print(f"[plot_tradetimeline] Keine validen Zeitwerte – Plot {filename} übersprungen.")
        return None
    plt.figure(figsize=(12,2))
    plt.scatter(series, [1]*len(series), marker='|', alpha=0.6)
    plt.title(title)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename
