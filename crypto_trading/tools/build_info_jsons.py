import json
import time
from pycoingecko import CoinGeckoAPI
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from indicator_db import TOP_INDICATORS, INDICATOR_INFOS

def build_crypto_infos(filename="crypto_infos.json"):
    try:
        top_n = int(input("Wie viele Coins (nach Marketcap) sollen abgefragt werden? [3-250]: ").strip())
        if not 3 <= top_n <= 250:
            print("Bitte gib eine Zahl zwischen 3 und 250 an!")
            return
    except Exception:
        print("Ungültige Eingabe!")
        return

    cg = CoinGeckoAPI()
    print(f"Lade Top {top_n} Kryptowährungen von CoinGecko ...\n")
    coins = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=top_n, page=1)
    info_dict = {}
    errors = []

    for idx, coin in enumerate(coins):
        coin_id = coin['id']
        try:
            bar = "[" + "="*int((idx+1)/top_n*30) + " "*(30-int((idx+1)/top_n*30)) + "]"
            print(f"{idx+1:3d}/{top_n} {bar}  {coin['symbol'].upper():<6}  ({coin['name']}) ...", end="\r", flush=True)
            def fetch_details():
                return cg.get_coin_by_id(coin_id, localization='de')
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fetch_details)
                try:
                    details = future.result(timeout=5)
                except FuturesTimeoutError:
                    print(f"\nTimeout bei {coin_id}, Coin wird übersprungen.")
                    errors.append((coin_id, "Timeout"))
                    continue
            bulletpoints_short = [
                f"Symbol: {details['symbol'].upper()}",
                f"Name: {details['name']}",
                f"Start: {details.get('genesis_date', 'unbekannt')}",
                f"Website: {details['links'].get('homepage', [''])[0]}",
                f"Marketcap: {details['market_data'].get('market_cap', {}).get('usd', 'n/a')}$",
                f"Preis: {details['market_data'].get('current_price', {}).get('usd', 'n/a')}$",
            ]
            desc = details['description']['de'] or details['description']['en'] or ""
            bullets_long = [x.strip() for x in desc.replace('\r', '\n').replace('. ', '.\n').split('\n') if x.strip()]
            bullets_long = bullets_long[:15] if bullets_long else ["Keine ausführliche Beschreibung verfügbar."]
            info_dict[details['symbol'].upper() + "USDT"] = {
                "name": details['name'],
                "shortinfo": bulletpoints_short,
                "longinfo": bullets_long,
                "website": details['links'].get('homepage', [''])[0],
                "launch": details.get('genesis_date', 'unbekannt'),
                "marketcap": details['market_data'].get('market_cap', {}).get('usd', 'n/a'),
            }
            time.sleep(0.85)
        except Exception as e:
            print(f"\nFehler bei {coin_id}: {e}")
            errors.append((coin_id, str(e)))
            continue
    print()
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=2)
    print(f"\n{filename} erfolgreich gespeichert ({len(info_dict)} Einträge)")
    if errors:
        print("\nFehler/Timeouts bei folgenden Coins:")
        for cid, err in errors:
            print(f"- {cid}: {err}")

def build_indicator_infos(filename="indicator_infos.json"):
    try:
        top_n = int(input(f"Wie viele Indikatoren (nach Effektivität) sollen angezeigt/gespeichert werden? [5-{min(100, len(TOP_INDICATORS))}]: ").strip())
        if not 1 <= top_n <= min(100, len(TOP_INDICATORS)):
            print(f"Bitte gib eine Zahl zwischen 1 und {min(100, len(TOP_INDICATORS))} an!")
            return
    except Exception:
        print("Ungültige Eingabe!")
        return

    ordered_keys = [i for i in TOP_INDICATORS if i in INDICATOR_INFOS][:top_n] + \
                   [i for i in INDICATOR_INFOS if i not in TOP_INDICATORS]
    indicator_infos_out = {k: INDICATOR_INFOS[k] for k in ordered_keys if k in INDICATOR_INFOS}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(indicator_infos_out, f, ensure_ascii=False, indent=2)
    print(f"\n{filename} erfolgreich gespeichert ({len(indicator_infos_out)} Einträge).")

    print("\n--- Indikatoren (Top-Effektivität zuerst) ---")
    print("{:<16} {:<40}".format("Kurzname", "Name"))
    print("-" * 57)
    for ind in indicator_infos_out:
        full = indicator_infos_out[ind].get("fullname", ind)
        print("{:<16} {:<40}".format(ind, full))
    print("\nDetails zu welchem Indikator anzeigen? (Kurzname eingeben, ENTER zum Überspringen)")
    ind = input("> ").strip()
    if ind and ind in indicator_infos_out:
        info = indicator_infos_out[ind]
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

if __name__ == "__main__":
    print("Starte Erstellung der Info-Dateien ...")
    build_crypto_infos()
    build_indicator_infos()
    print("\nFertig! Beide JSON-Dateien wurden erstellt.")
