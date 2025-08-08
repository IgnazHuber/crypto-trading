# mod_png.py
import matplotlib.pyplot as plt
from .portfolio_metrics import calculate_portfolio_metrics

def export_png_equity(file_label, equity_df, output_path=None):
    output_path = output_path or f"results/equity_{file_label}.png"
    fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    ax[0].plot(equity_df["Time"], equity_df["Equity"], label="Equity")
    ax[0].grid(True)
    ax[0].legend()
    ax[1].plot(equity_df["Time"], equity_df["Drawdown"], color="red", label="Drawdown")
    ax[1].grid(True)
    ax[1].legend()

    # kleine KPI-Box
    metrics = calculate_portfolio_metrics({file_label: equity_df.set_index("Time")["Equity"]})
    txt = "\n".join([f"{r.Asset}: Sharpe {r.Sharpe:.2f}, MaxDD {r.MaxDrawdown:.2%}" for r in metrics.itertuples()])
    ax[0].text(0.02, 0.95, txt, transform=ax[0].transAxes,
               fontsize=8, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[PNG] Equity/Drawdown gespeichert: {output_path}")
