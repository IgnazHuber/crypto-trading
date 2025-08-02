import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).parent / "candlestick_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Alle Parquet-Dateien finden
parquet_files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))

# 2. Übersicht: Alle gefundenen Dateien parsen
overview = []
for f in parquet_files:
    fname = os.path.basename(f)
    parts = fname.replace(".parquet", "").split("_")
    if len(parts) < 3:
        continue
    # Beispiele: BTCUSDT_1d_1year_ccxt.parquet, BTCUSDT_max_1d_5years.parquet
    symbol = parts[0]
    freq = parts[1]
    if "max" in freq:
        freq = parts[2]  # max_1d_5years → 1d
        period = parts[3] if len(parts) > 3 else "unknown"
    else:
        period = parts[2]
    overview.append({"file": f, "symbol": symbol, "freq": freq, "period": period})

overview_df = pd.DataFrame(overview)
if overview_df.empty:
    print("Keine Parquet-Dateien gefunden im Verzeichnis:", DATA_DIR)
    exit()

# 3. Übersichtstabelle anzeigen
print("\nVerfügbare Daten:")
print(overview_df[["symbol", "period", "freq"]].drop_duplicates().to_string(index=False))

# 4. Nutzer-Auswahl: Symbol
symbol_choices = sorted(overview_df["symbol"].unique())
print("\nWelche Krypto soll geplottet werden?")
for i, s in enumerate(symbol_choices):
    print(f"{i+1:2d}. {s}")
idx = int(input("Bitte Nummer eingeben: ")) - 1
symbol = symbol_choices[idx]

# 5. Alle Zeitfenster/Frequenzen für dieses Symbol plotten
sub_df = overview_df[overview_df["symbol"] == symbol]

for _, row in sub_df.iterrows():
    df = pd.read_parquet(row["file"])
    # Optional: Spaltennamen normalisieren
    df.columns = [c.lower() for c in df.columns]
    # Candlestick nur, wenn alle OHLC vorhanden
    if not all(c in df.columns for c in ["open", "high", "low", "close"]):
        print(f"Überspringe (fehlende Spalten): {row['file']}")
        continue
    df_plot = df.copy()
    df_plot.index.name = "Date"

    title = f"{symbol} | Zeitraum: {row['period']} | Frequenz: {row['freq']}"
    print("Plot:", title)
    out_path = OUTPUT_DIR / f"{symbol}_{row['period']}_{row['freq']}_candle.png"
    # mplfinance erwartet OHLC-Format, index = datetime
    try:
        mpf.plot(df_plot, type='candle', title=title, style='charles',
                 mav=(10, 20), savefig=str(out_path), volume=True)
    except Exception as e:
        print(f"Fehler beim Plotten {title}: {e}")

print(f"\nAlle Charts gespeichert in {OUTPUT_DIR.resolve()}")
