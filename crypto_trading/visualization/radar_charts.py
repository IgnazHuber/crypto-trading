import os
import numpy as np
import matplotlib.pyplot as plt
from crypto_trading.indicators import INDICATORS

def create_radarplots_for_trade(trade, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Dynamisch: alle Indikatorwerte
    values = [float(trade.get(ind, 0)) for ind in INDICATORS]
    arr = np.array(values, dtype=np.float64)
    if np.nanmax(arr) - np.nanmin(arr) > 0:
        values_norm = ((arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))).tolist()
    else:
        values_norm = [0.5 for _ in arr]
    radar_plot(values, INDICATORS, os.path.join(save_dir, "radar_raw.png"), "Radar (Rohwerte)")
    radar_plot(values_norm, INDICATORS, os.path.join(save_dir, "radar_norm.png"), "Radar (Normiert)")

def radar_plot(values, labels, path, title):
    n = len(labels)
    values_plot = list(values) + [values[0]]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_plot = list(angles) + [angles[0]]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles_plot, values_plot, "o-", linewidth=2)
    ax.fill(angles_plot, values_plot, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles), labels)
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
