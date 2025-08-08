import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta

# Funktion zum Abrufen von historischen Kursdaten von Binance
def fetch_binance_data(symbol, interval, start_time, end_time):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000  # Maximaler Datenpunkt pro Anfrage
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

# Funktion zum Konvertieren von Daten in ein DataFrame und Speichern im Parquet-Format
def save_to_parquet(data, filename):
    # Konvertiere die Daten in ein DataFrame
    df = pd.DataFrame(data, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                                      'Close Time', 'Quote Asset Volume', 'Number of Trades', 
                                      'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
    
    # Konvertiere Zeitstempel in lesbares Format
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

    # Speichere das DataFrame im Parquet-Format
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filename)

# Hauptprogramm
if __name__ == "__main__":
    symbol = "BTCUSDT"  # Beispiel: Bitcoin gegen USDT
    interval = "1s"  # 1 Sekunde
    end_time = int(datetime.now().timestamp() * 1000)  # Aktuelle Zeit in Millisekunden
    start_time = end_time - (60 * 1000)  # Letzte Minute als Beispiel

    # Abrufen der Daten
    data = fetch_binance_data(symbol, interval, start_time, end_time)
    
    # Speichern der Daten im Parquet-Format
    save_to_parquet(data, 'historical_data.parquet')

    print("Daten erfolgreich gespeichert!")
