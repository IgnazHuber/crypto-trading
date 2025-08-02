# tools/build_indicator_infos_and_show.py

import json
import os

INDICATOR_JSON_PATH = "indicator_infos.json"

# --- Beispielstruktur für Top-Indikatoren, beliebig erweiterbar ---
INDICATOR_INFOS = {
    "MACD": {
        "shortname": "MACD",
        "fullname": "Moving Average Convergence Divergence",
        "shortinfo": [
            "Trendfolge-Indikator.",
            "Zwei exponentielle gleitende Durchschnitte.",
            "Identifiziert Trendwenden.",
            "MACD-Linie, Signallinie, Histogramm.",
            "Beliebt im mittelfristigen Trading."
        ],
        "market_env": [
            "Funktioniert in trendstarken Märkten.",
            "Fehlsignale in Seitwärtsphasen.",
            "Mit Volumen-Bestätigung sicherer.",
            "Für Krypto und Aktien geeignet.",
            "Mit RSI und BB kombinierbar."
        ],
        "typical_values": [
            "Standard: 12/26/9 EMA.",
            "Cross Signal: MACD kreuzt Signallinie.",
            "Histogramm als Trendstärke.",
            "Null-Linie als Trendfilter."
        ],
        "combination": [
            "Sinnvoll mit RSI, ADX.",
            "Mit Volumenindikatoren.",
            "Nicht mit weiteren Trendfolgern alleine.",
            "Mit Candlestick-Mustern.",
            "Für Gewinnmitnahmen (Exit) nutzen."
        ],
        "longinfo": [
            "Gerald Appel entwickelte den MACD in den 1970er Jahren.",
            "MACD = EMA(12) – EMA(26).",
            "Signal-Linie = EMA(9) des MACD.",
            "Histogramm = MACD – Signallinie.",
            "Positive Werte → Aufwärtstrend.",
            "Negative Werte → Abwärtstrend.",
            "Divergenzen als Trendumkehr-Signal.",
            "Beliebt für mittelfristiges Swing-Trading.",
            "Nicht normiert – Werte sind relativ.",
            "Histogramm zeigt Trenddynamik.",
            "Mit Volumen-Filter besonders robust.",
            "Vorsicht bei extremer Volatilität.",
            "Backtestbar in allen Märkten.",
            "Signal-Cross als Kauf-/Verkaufssignal.",
            "Starke Trends liefern beste MACD-Signale."
        ]
    },
    "RSI": {
        "shortname": "RSI",
        "fullname": "Relative Strength Index",
        "shortinfo": [
            "Oszillator für Marktstärke.",
            "Typisch 14 Perioden.",
            "Werte 0 bis 100.",
            "Überkauft >70, überverkauft <30.",
            "Für Trendumkehr-Signale."
        ],
        "market_env": [
            "Sinnvoll in trendlosen oder schwachen Märkten.",
            "Warnsignal bei starken Trends.",
            "Mit Trendfiltern kombinieren.",
            "Hilfreich mit SMA/EMA.",
            "Nicht als alleiniges Signal."
        ],
        "typical_values": [
            "Standard: 14 Perioden.",
            "Überkauft: >70.",
            "Überverkauft: <30.",
            "Neutral: 40–60."
        ],
        "combination": [
            "Mit MACD, SMA, BB.",
            "Volumenindikatoren empfohlen.",
            "Nicht nur mit anderen Oszillatoren.",
            "Mit Trendfiltern robust.",
            "Filter für Entry/Exit."
        ],
        "longinfo": [
            "J. Welles Wilder entwickelte den RSI 1978.",
            "100 – (100 / (1 + RS)), RS = Gewinn/Verlust.",
            "Nahe 100: überkauft, nahe 0: überverkauft.",
            "Kann lange im Extrembereich bleiben.",
            "Divergenzen als Warnsignal.",
            "Glättung über gleitende Durchschnitte möglich.",
            "Empfindlich auf sehr kurze Perioden.",
            "Kombination mit weiteren Indikatoren.",
            "Für Konsolidierungsphasen nützlich.",
            "RSI weltweit anerkannt.",
            "Signalzonen können je nach Markt angepasst werden.",
            "Mehrere Zeitrahmen machen das Signal robuster.",
            "Nicht mit Stochastik verwechseln.",
            "Fehlsignale in starken Trends.",
            "Hilfreich für Rebound-Strategien."
        ]
    },
    "BollingerBands": {
        "shortname": "BB",
        "fullname": "Bollinger Bands",
        "shortinfo": [
            "Volatilitätsindikator.",
            "SMA + 2 Bänder (±2 Stdabw.).",
            "Expansion/Verengung zeigt Marktphasen.",
            "Oberband = Überkauft, Unterband = Überverkauft.",
            "Ideal für Range- und Ausbruchsstrategien."
        ],
        "market_env": [
            "Volatilitäts-Cluster zeigen sich im Band.",
            "Seitwärtsmärkte: viele Bandkontakte.",
            "Trendmärkte: Bandbruch als Signal.",
            "Enge Bänder: Bewegung steht bevor.",
            "Nützlich bei Rebound-Strategien."
        ],
        "typical_values": [
            "Standard: 20er SMA, ±2 Stdabw.",
            "Oberband: Überkauft.",
            "Unterband: Überverkauft.",
            "Bandbreite als Volatilitätsmaß."
        ],
        "combination": [
            "Mit RSI, MACD kombinieren.",
            "Mit Volumenindikatoren sinnvoll.",
            "Nicht allein für Trendfilter.",
            "Mit Candlestick-Mustern robust.",
            "Stochastik als zusätzlicher Filter."
        ],
        "longinfo": [
            "John Bollinger entwickelte die Bänder 1980er.",
            "Zeigen aktuelle Volatilität.",
            "Enge Bänder → Ausbruch steht bevor.",
            "Bandüberschreitung = Trendchance.",
            "Bandrand = Mean-Reversion-Möglichkeit.",
            "Kein alleiniges Einstiegssignal.",
            "Viele Fehlsignale bei wenig Volumen.",
            "Mit Trendfilter sehr robust.",
            "Standardwerte anpassbar.",
            "Bandbreite signalisiert Volatilität.",
            "Backtestbar in allen Märkten.",
            "Auch für Aktien und Forex geeignet.",
            "Range-Trading optimal darstellbar.",
            "Weltweit anerkannter Indikator.",
            "Sehr visualisierungsstark."
        ]
    }
    # Ergänze hier beliebig viele weitere Indikatoren
}

TOP_INDICATORS = [
    "MACD", "RSI", "BollingerBands"
    # ... hier nach Effektivität geordnet ergänzen
]

def build_indicator_infos(filename=INDICATOR_JSON_PATH):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(INDICATOR_INFOS, f, ensure_ascii=False, indent=2)
    print(f"\n{filename} erfolgreich gespeichert ({len(INDICATOR_INFOS)} Einträge).")

def print_indicator_overview(indicator_infos):
    ordered = [i for i in TOP_INDICATORS if i in indicator_infos] + \
              [i for i in indicator_infos if i not in TOP_INDICATORS]
    print("\n--- Indikatoren (Top-Effektivität zuerst) ---")
    print("{:<16} {:<40}".format("Kurzname", "Name"))
    print("-" * 57)
    for ind in ordered:
        full = indicator_infos[ind].get("fullname", ind)
        print("{:<16} {:<40}".format(ind, full))
    print("\n")

def show_indicator_details(indicator_infos):
    print("Details zu welchem Indikator anzeigen? (Kurzname eingeben, ENTER zum Überspringen)")
    ind = input("> ").strip()
    if not ind or ind not in indicator_infos:
        print("Abbruch oder Indikator nicht gefunden.")
        return
    info = indicator_infos[ind]
    print(f"\n==== {ind} | {info.get('fullname', '')} ====")
    print("\nKurzinfo:")
    for bp in info.get("shortinfo", []):
        print("  -", bp)
    print("\nMarktumfeld:")
    for bp in info.get("market_env", []):
        print("  •", bp)
    print("\nTypische Werte/Grenzwerte:")
    for bp in info.get("typical_values", []):
        print("  •", bp)
    print("\nSinnvolle Kombinationen:")
    for bp in info.get("combination", []):
        print("  •", bp)
    print("\nLanginfo:")
    for bp in info.get("longinfo", []):
        print("  •", bp)
    print("\n")

def main():
    # Erzeuge und lade JSON
    build_indicator_infos()
    with open(INDICATOR_JSON_PATH, encoding="utf-8") as f:
        indicator_infos = json.load(f)
    # Übersicht
    print_indicator_overview(indicator_infos)
    # Details (Menü)
    show_indicator_details(indicator_infos)

if __name__ == "__main__":
    main()
