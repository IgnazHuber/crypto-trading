# optimizer_visuals.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_optimization_results(path):
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError("Dateiformat nicht unterstützt (nur .xlsx oder .csv)")

def plot_heatmap(df, output_path="results/heatmap.png"):
    """
    Heatmap: Endkapital vs. Maximaler Einzelverlust.
    """
    pivot = df.pivot_table(values="end_capital", index="max_single_loss", aggfunc="mean")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Heatmap: Endkapital vs. Maximaler Einzelverlust")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Visual] Heatmap gespeichert: {output_path}")

def plot_scatter(df, output_path="results/scatter.png"):
    """
    Scatterplot: Endkapital vs. Maximaler Einzelverlust (Farbcodiert nach Trades).
    """
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df["max_single_loss"], df["end_capital"],
                     c=df["trades"], cmap="viridis", alpha=0.7)
    plt.colorbar(sc, label="Trades")
    plt.xlabel("Maximaler Einzelverlust")
    plt.ylabel("Endkapital")
    plt.title("Scatter: Endkapital vs. Max. Einzelverlust")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Visual] Scatter gespeichert: {output_path}")

def top10_table(df, output_path="results/top10.xlsx"):
    """
    Speichert Top 10 Strategien mit allen Parametern.
    """
    top10 = df.sort_values(["end_capital", "max_single_loss"], ascending=[False, True]).head(10)
    top10.to_excel(output_path, index=False)
    print(f"[Visual] Top 10 Strategien gespeichert: {output_path}")
    return top10

def create_all_visuals(result_file):
    df = load_optimization_results(result_file)
    os.makedirs("results", exist_ok=True)
    plot_heatmap(df)
    plot_scatter(df)
    top10_table(df)
    print("[Visual] Alle Diagramme erstellt.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Verwendung: python -m crypto_trading.optimizer_visuals <pfad_zur_optimierungsergebnisdatei>")
    else:
        create_all_visuals(sys.argv[1])
