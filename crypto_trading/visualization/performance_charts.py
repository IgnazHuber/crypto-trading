import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_top_flop_bars(top5, flop5, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if top5:
        df = pd.DataFrame(top5)
        plt.figure(figsize=(5, 3))
        plt.bar(df["symbol"], df["pnl_abs"], color="green")
        plt.title("Top 5 Trades")
        plt.ylabel("PnL abs")
        plt.tight_layout()
        path = os.path.join(out_dir, "top5.png")
        plt.savefig(path)
        plt.close()
    if flop5:
        df = pd.DataFrame(flop5)
        plt.figure(figsize=(5, 3))
        plt.bar(df["symbol"], df["pnl_abs"], color="red")
        plt.title("Flop 5 Trades")
        plt.ylabel("PnL abs")
        plt.tight_layout()
        path = os.path.join(out_dir, "flop5.png")
        plt.savefig(path)
        plt.close()
def plot_monthly_performance(trades_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if trades_df.empty:
        return
    df = trades_df.copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    # pnl_abs neu berechnen
    if "pnl_abs" not in df.columns:
        df["pnl_abs"] = df["exit_price"] - df["entry_price"]
    df["month"] = df["exit_date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")["pnl_abs"].sum()

