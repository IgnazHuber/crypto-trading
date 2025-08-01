import os
import matplotlib.pyplot as plt

def plot_trades(df, entries, exits, short_entries, short_exits, save_path):
    """
    Zeichnet den Kursverlauf und markiert die Entry/Exit Punkte.
    Speichert als PNG.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="Close", color="blue")

    # Long Trades
    if entries is not None and exits is not None:
        ax.scatter(entries.index[entries], df["Close"][entries],
                   marker="^", color="green", label="Long Entry")
        ax.scatter(exits.index[exits], df["Close"][exits],
                   marker="v", color="red", label="Long Exit")

    # Short Trades (falls vorhanden)
    if short_entries is not None and short_exits is not None:
        ax.scatter(short_entries.index[short_entries], df["Close"][short_entries],
                   marker="v", color="orange", label="Short Entry")
        ax.scatter(short_exits.index[short_exits], df["Close"][short_exits],
                   marker="^", color="purple", label="Short Exit")

    ax.legend()
    ax.set_title("Trades")
    ax.grid(True)

    # Immer PNG erzwingen
    root, ext = os.path.splitext(save_path)
    if ext.lower() not in [".png", ".jpg", ".jpeg"]:
        save_path = root + ".png"

    plt.savefig(save_path, format="png")
    plt.close(fig)
