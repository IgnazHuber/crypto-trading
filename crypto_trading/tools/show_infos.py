import json
import os

# ---- Konfiguration ----
CRYPTO_JSON_PATH = "crypto_infos.json"
INDICATOR_JSON_PATH = "indicator_infos.json"

# Manuelle Top-Order für Indikatoren (anpassen nach Gusto oder Research!)
TOP_INDICATORS = [
    "MACD", "RSI", "EMA", "SMA", "BollingerBands", "ADX", "OBV", "VWAP", "CCI", "ATR",
    "Stochastic", "MFI", "Williams %R", "Ichimoku", "Parabolic SAR", "ROC", "TRIX", "Chaikin Oscillator"
]
# Rest alphabetisch ergänzen, wenn nötig.

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def print_crypto_overview(crypto_infos):
    # Sortiert nach Marketcap (sofern vorhanden)
    coins = [
        (symbol, data["name"], data.get("marketcap", 0), data.get("shortinfo", []))
        for symbol, data in crypto_infos.items()
    ]
    coins = sorted(coins, key=lambda x: float(x[2]) if x[2] not in ["n/a", None, ""] else 0, reverse=True)
    print("\n--- Kryptowährungen (sortiert nach Marktkapitalisierung) ---")
    print("{:<10} {:<22} {:>16}".format("Symbol", "Name", "Marketcap (USD)"))
    print("-" * 55)
    for sym, name, mc, _ in coins:
        print("{:<10} {:<22} {:>16,.0f}".format(sym, name, float(mc) if mc not in ["n/a", None, ""] else 0))
    print("\n")

def show_crypto_details(crypto_infos):
    print("Details zu welchem Coin anzeigen? (Ticker eingeben, z.B. BTCUSDT, ENTER zum Überspringen)")
    symbol = input("> ").strip().upper()
    if not symbol or symbol not in crypto_infos:
        print("Abbruch oder Coin nicht gefunden.")
        return
    info = crypto_infos[symbol]
    print(f"\n==== {symbol} | {info['name']} ====")
    print("\nKurzinfo:")
    for bp in info["shortinfo"]:
        print("  -", bp)
    print("\nLanginfo:")
    for bp in info["longinfo"]:
        print("  •", bp)
    print(f"\nWebsite: {info.get('website', 'n/a')} | Start: {info.get('launch', 'n/a')} | Marketcap: {info.get('marketcap', 'n/a')}\n")

def print_indicator_overview(indicator_infos):
    # Effektivitäts-Sortierung: erst TOP_INDICATORS, dann Rest alphabetisch
    ind_list = []
    for key in indicator_infos.keys():
        ind_list.append(key)
    ordered = [i for i in TOP_INDICATORS if i in ind_list]
    ordered += sorted(set(ind_list) - set(ordered))
    print("\n--- Indikatoren (Top-Effektivität zuerst) ---")
    print("{:<16} {:<40}".format("Kurzname", "Name"))
    print("-" * 57)
    for ind in ordered:
        full = indicator_infos[ind].get("fullname", ind)
        print("{:<16} {:<40}".format(ind, full))
    print("\n")

def show_indicator_details(indicator_infos):
    print("Details zu welchem Indikator anzeigen? (Kurzname eingeben, z.B. MACD, ENTER zum Überspringen)")
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
    if not os.path.isfile(CRYPTO_JSON_PATH):
        print("Krypto-Info-File fehlt! Bitte erst generieren.")
        return
    if not os.path.isfile(INDICATOR_JSON_PATH):
        print("Indikator-Info-File fehlt! Bitte erst generieren.")
        return
    crypto_infos = load_json(CRYPTO_JSON_PATH)
    indicator_infos = load_json(INDICATOR_JSON_PATH)

    print_crypto_overview(crypto_infos)
    show_crypto_details(crypto_infos)
    print_indicator_overview(indicator_infos)
    show_indicator_details(indicator_infos)

if __name__ == "__main__":
    main()
