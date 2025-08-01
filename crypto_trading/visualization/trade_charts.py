import matplotlib.pyplot as plt
import os

def plot_trades(price_df, trades_df, symbol, out_path="trades.png"):
    """
    Zeichnet Kursverlauf mit Entry- und Exit-Markierungen.
    """
    if price_df.empty:
        print("Keine Preisdaten, Chart wird nicht erstellt.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(price_df.index, price_df["Close"], label="Close", color="blue")

    if not trades_df.empty:
        for _, t in trades_df.iterrows():
            entry_date = t["entry_date"]
            exit_date = t["exit_date"]
            entry_price = t["entry_price"]
            exit_price = t["exit_price"]

            ax.scatter(entry_date, entry_price, color="green", marker="^", s=100, label="Entry")
            ax.scatter(exit_date, exit_price, color="red", marker="v", s=100, label="Exit")
            ax.annotate("Entry", (entry_date, entry_price), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color="green")
            ax.annotate("Exit", (exit_date, exit_price), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=8, color="red")

    ax.set_title(f"Trades für {symbol}")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Preis")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Trades-Chart gespeichert: {os.path.abspath(out_path)}")
