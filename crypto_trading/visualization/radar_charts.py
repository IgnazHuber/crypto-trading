# crypto_trading/visualization/radar_charts.py
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_radar_chart(indicators: dict, title: str, out_path: str):
    """
    Zeichnet ein Radar-Chart für gegebene Indikatorwerte.
    indicators: dict {Indikatorname: Wert}
    """
    labels = list(indicators.keys())
    values = list(indicators.values())
    num_vars = len(labels)

    # Werte schließen (Zyklus)
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title, y=1.1)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Radar-Chart gespeichert: {os.path.abspath(out_path)}")

def plot_trade_radar(trade: dict, out_dir: str):
    """
    Erzeugt Radar-Charts (roh und normiert) für einen Trade.
    Erwartet trade mit Indikatorwerten (float).
    """
    os.makedirs(out_dir, exist_ok=True)
    indicators = {k: trade[k] for k in trade.keys() if k not in 
                  ["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "size"]}

    # Rohwerte
    plot_radar_chart(indicators, "Indikatoren (Rohwerte)", os.path.join(out_dir, "radar_raw.png"))

    # Normierung 0…1
    vals = np.array(list(indicators.values()), dtype=float)
    if np.nanmax(vals) - np.nanmin(vals) > 0:
        vals_norm = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals))
    else:
        vals_norm = np.ones_like(vals) * 0.5
    indicators_norm = dict(zip(indicators.keys(), vals_norm))
    plot_radar_chart(indicators_norm, "Indikatoren (Normiert)", os.path.join(out_dir, "radar_norm.png"))
