import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_trades(df, entries, exits, short_entries, short_exits, save_path):
    """
    Zeichnet den Kursverlauf und markiert Entry/Exit-Punkte für Long/Short-Trades.
    Speichert als PNG. Akzeptiert Signale als np.ndarray, list oder pd.Series.
    """
    entries = pd.Series(entries, index=df.index) if not isinstance(entries, pd.Series) else entries
    exits = pd.Series(exits, index=df.index) if not isinstance(exits, pd.Series) else exits
    short_entries = pd.Series(short_entries, index=df.index) if not isinstance(short_entries, pd.Series) else short_entries
    short_exits = pd.Series(short_exits, index=df.index) if not isinstance(short_exits, pd.Series) else short_exits

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="Close", color="blue", linewidth=1.4)

    # Long Trades
    if entries.any():
        ax.scatter(entries.index[entries], df["Close"][entries], marker="^", color="green", label="Long Entry", zorder=5)
    if exits.any():
        ax.scatter(exits.index[exits], df["Close"][exits], marker="v", color="red", label="Long Exit", zorder=5)
    # Short Trades
    if short_entries.any():
        ax.scatter(short_entries.index[short_entries], df["Close"][short_entries], marker="x", color="orange", label="Short Entry", zorder=5)
    if short_exits.any():
        ax.scatter(short_exits.index[short_exits], df["Close"][short_exits], marker="d", color="purple", label="Short Exit", zorder=5)

    ax.set_title("Trades", fontsize=15)
    ax.set_xlabel("Zeit", fontsize=12)
    ax.set_ylabel("Preis", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
