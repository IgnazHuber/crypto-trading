import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_visuals(df, html_path):
    """
    Erstellt interaktive Visualisierung der Optimierungsergebnisse.
    Zeigt:
      - Scatterplot (Endkapital vs. Maximalverlust)
      - Heatmap (Verteilung Endkapital nach Parametern)
    Speichert eine kombinierte HTML-Datei.
    """
    # Scatterplot
    scatter = px.scatter(
        all_results_df,
        x="end_capital",
        y="max_single_loss",
        color="context",
        hover_data=["params"]
    )


    # Heatmap-Vorbereitung: Beispiel mit ema_length und sma_length
    if "ema_length" in df["params"].iloc[0] and "sma_length" in df["params"].iloc[0]:
        df_heatmap = pd.DataFrame([
            {**p, "end_capital": ec} for p, ec in zip(df["params"], df["end_capital"])
        ])
        heatmap = px.density_heatmap(
            df_heatmap,
            x="ema_length",
            y="sma_length",
            z="end_capital",
            title="Heatmap: Endkapital (ema_length vs sma_length)",
            nbinsx=len(df_heatmap["ema_length"].unique()),
            nbinsy=len(df_heatmap["sma_length"].unique())
        )
    else:
        heatmap = go.Figure()

    # Kombinieren
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(scatter.to_html(full_html=False, include_plotlyjs="cdn"))
        if heatmap:
            f.write("<hr>")
            f.write(heatmap.to_html(full_html=False, include_plotlyjs=False))
    print(f"[Visualizer] HTML gespeichert: {html_path}")
