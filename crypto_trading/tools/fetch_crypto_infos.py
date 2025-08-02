from pycoingecko import CoinGeckoAPI
import json
import time

cg = CoinGeckoAPI()
coins = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=100, page=1)
info_dict = {}

for coin in coins:
    coin_id = coin['id']
    details = cg.get_coin_by_id(coin_id, localization='de')
    bulletpoints_short = [
        f"Symbol: {details['symbol'].upper()}",
        f"Name: {details['name']}",
        f"Start: {details.get('genesis_date', 'unbekannt')}",
        f"Website: {details['links'].get('homepage', [''])[0]}",
        f"Marketcap: {details['market_data'].get('market_cap', {}).get('usd', 'n/a')}$",
        f"Preis: {details['market_data'].get('current_price', {}).get('usd', 'n/a')}$",
    ]
    desc = details['description']['de'] or details['description']['en'] or ""
    # Für die Langinfo splitten wir in Sätze/Bulletpoints:
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
    # API ist freundlich, aber langsamer = weniger Rate-Limit-Probleme
    time.sleep(0.9)

with open("crypto_infos.json", "w", encoding="utf-8") as f:
    json.dump(info_dict, f, ensure_ascii=False, indent=2)

print("crypto_infos.json erfolgreich gespeichert (Top 100)")
