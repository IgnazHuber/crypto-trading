"""
Modul: plots_kpi.py

Erzeugt wissenschaftliche Visualisierungen für Einzelstrategien, Paramsets und Portfolios:
- Equity, Drawdown, Rolling KPIs, PnL-Histogramm, Monats-Heatmap, Outlier-Boxplot,
  Sharpe-vs-PnL-Scatter, Portfolio-Allokation, Korrelationsmatrix, Trade-Zeitstrahl.

Alle Funktionen speichern PNGs und geben den Dateipfad zurück.

Funktionen:
- plot_equity_curve
- plot_drawdown
- plot_rolling_sharpe
- plot_pnl_hist
- plot_month_heatmap
- plot_sharpe_pnl_scatter
- plot_portfolio_donut
- plot_pnl_box
- plot_corr_matrix
- plot_tradetimeline

Author: ChatGPT Research, 2025
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

def plot_equity_curve(eq, filename, title="Equity Curve"):
    plt.figure(figsize=(10, 3))
    eq.plot(title=title)
    plt.ylabel("Kapital")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_drawdown(eq, filename, title="Drawdown-Kurve"):
    drawdown = (eq / eq.cummax() - 1)
    plt.figure(figsize=(10,3))
    plt.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.4)
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_rolling_sharpe(eq, filename, window=500, freq_factor=24*365, title=None):
    returns = eq.pct_change().dropna()
    rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(freq_factor)
    plt.figure(figsize=(10,3))
    plt.plot(rolling_sharpe)
    plt.title(title or f"Rolling Sharpe ({window} Perioden)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_pnl_hist(trades, filename, title="PnL-Histogramm (Einzeltrades)"):
    plt.figure(figsize=(7,3))
    plt.hist(trades['gewinn_verlust'].dropna(), bins=30, alpha=0.7)
    plt.title(title)
    plt.xlabel("Gewinn/Verlust [%]")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_month_heatmap(trades, filename, title="Monats-Performance Heatmap"):
    if trades.empty: return None
    df = trades.copy()
    df['year'] = pd.to_datetime(df['entry_time']).dt.year
    df['month'] = pd.to_datetime(df['entry_time']).dt.month
    heat = df.groupby(['year','month'])['gewinn_verlust'].sum().unstack().fillna(0)
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
    plt.figure(figsize=(7,4))
    plt.scatter(perf_df['Sharpe'], perf_df['Gesamt-PnL'], alpha=0.7)
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Gesamt-PnL [%]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_portfolio_donut(df_all_trades, filename, title="Portfolio-Allokation nach Asset (PnL)"):
    sizes = df_all_trades.groupby('asset')['gewinn_verlust'].sum()
    plt.figure(figsize=(5,5))
    plt.pie(sizes, labels=sizes.index, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_pnl_box(trades, filename, title="Outlier/Boxplot aller Gewinne/Verluste"):
    plt.figure(figsize=(7,3))
    plt.boxplot(trades['gewinn_verlust'].dropna(), vert=False)
    plt.title(title)
    plt.xlabel("Gewinn/Verlust [%]")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_corr_matrix(perf_df, filename, metric="Gesamt-PnL", title="Korrelation zwischen Assets"):
    if len(perf_df['Asset'].unique()) < 2:
        return None
    piv = perf_df.pivot(index='param_idx', columns='Asset', values=metric)
    corr = piv.corr()
    plt.figure(figsize=(7,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def plot_tradetimeline(trades, filename, title="Zeitstrahl: Wann wurde investiert?"):
    df = trades.copy()
    plt.figure(figsize=(12,2))
    plt.scatter(pd.to_datetime(df['entry_time']), [1]*len(df), marker='|', alpha=0.6)
    plt.title(title)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

# Optionaler Self-Test
if __name__ == "__main__":
    print("plots_kpi.py – Self-Test – bitte als Modul importieren!")
