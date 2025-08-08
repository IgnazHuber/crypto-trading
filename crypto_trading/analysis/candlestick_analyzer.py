import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import logging
import tempfile
from datetime import datetime, timedelta
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from tqdm import tqdm
import warnings

# Unterdrücke Plotly-Warnungen
warnings.filterwarnings("ignore", category=UserWarning)

# Logging einrichten
try:
    logging.basicConfig(
        filename=os.path.join('results', 'candlestick_analyzer.log'),
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
except Exception as e:
    print(f"Fehler beim Einrichten des Loggings: {str(e)}")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Funktion zur Erkennung von Candlestick-Formationen
def detect_candlestick_patterns(df):
    logging.debug(f"Starte Candlestick-Analyse für DataFrame mit {len(df)} Zeilen")
    patterns = []
    
    for i in tqdm(range(2, len(df)), desc="Analyse Candlestick-Formationen", leave=False):
        prev2 = df.iloc[i-2]
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        
        pattern = {
            'timestamp': curr['timestamp'],
            'pattern': None,
            'signal': None,
            'reason': None
        }
        
        body_size = lambda x: abs(x['close'] - x['open'])
        is_bullish = lambda x: x['close'] > x['open']
        is_bearish = lambda x: x['close'] < x['open']
        
        if (is_bearish(prev) and is_bullish(curr) and
            curr['open'] < prev['close'] and curr['close'] > prev['open']):
            pattern['pattern'] = 'Bullish Engulfing'
            pattern['signal'] = 'Kaufen'
            pattern['reason'] = 'Steigender Kurs nach bärischer Kerze, signalisiert Umkehr nach oben'
        
        elif (is_bullish(prev) and is_bearish(curr) and
              curr['open'] > prev['close'] and curr['close'] < prev['open']):
            pattern['pattern'] = 'Bearish Engulfing'
            pattern['signal'] = 'Verkaufen'
            pattern['reason'] = 'Fallender Kurs nach bullischer Kerze, signalisiert Umkehr nach unten'
        
        elif (is_bullish(curr) and
              (curr['open'] - curr['low']) > 2 * body_size(curr) and
              (curr['high'] - curr['close']) < body_size(curr)):
            pattern['pattern'] = 'Hammer'
            pattern['signal'] = 'Kaufen'
            pattern['reason'] = 'Langer unterer Schatten, signalisiert mögliche Umkehr nach oben'
        
        elif (is_bearish(curr) and
              (curr['high'] - curr['open']) > 2 * body_size(curr) and
              (curr['close'] - curr['low']) < body_size(curr)):
            pattern['pattern'] = 'Shooting Star'
            pattern['signal'] = 'Verkaufen'
            pattern['reason'] = 'Langer oberer Schatten, signalisiert mögliche Umkehr nach unten'
        
        elif body_size(curr) < 0.1 * (curr['high'] - curr['low']):
            pattern['pattern'] = 'Doji'
            pattern['signal'] = 'Neutral'
            pattern['reason'] = 'Unentschlossenheit im Markt, mögliche Trendumkehr oder Fortsetzung'
        
        elif (is_bearish(prev2) and body_size(prev) < 0.3 * (prev['high'] - prev['low']) and
              is_bullish(curr) and curr['close'] > prev2['open'] * 0.5):
            pattern['pattern'] = 'Morning Star'
            pattern['signal'] = 'Kaufen'
            pattern['reason'] = 'Dreikerzenmuster, signalisiert Umkehr nach oben nach Abwärtstrend'
        
        elif (is_bullish(prev2) and body_size(prev) < 0.3 * (prev['high'] - prev['low']) and
              is_bearish(curr) and curr['close'] < prev2['open'] * 0.5):
            pattern['pattern'] = 'Evening Star'
            pattern['signal'] = 'Verkaufen'
            pattern['reason'] = 'Dreikerzenmuster, signalisiert Umkehr nach unten nach Aufwärtstrend'
        
        elif (is_bearish(prev) and is_bullish(curr) and
              curr['open'] > prev['close'] and curr['close'] < prev['open']):
            pattern['pattern'] = 'Bullish Harami'
            pattern['signal'] = 'Kaufen'
            pattern['reason'] = 'Kleine bullische Kerze innerhalb einer großen bärischen, signalisiert Umkehr'
        
        elif (is_bullish(prev) and is_bearish(curr) and
              curr['open'] < prev['close'] and curr['close'] > prev['open']):
            pattern['pattern'] = 'Bearish Harami'
            pattern['signal'] = 'Verkaufen'
            pattern['reason'] = 'Kleine bärische Kerze innerhalb einer großen bullischen, signalisiert Umkehr'
        
        elif (is_bearish(prev) and is_bullish(curr) and
              curr['open'] < prev['low'] and curr['close'] > prev['open'] * 0.5):
            pattern['pattern'] = 'Piercing Line'
            pattern['signal'] = 'Kaufen'
            pattern['reason'] = 'Bullische Kerze durchbricht bärische Kerze, signalisiert Umkehr nach oben'
        
        if pattern['pattern']:
            patterns.append(pattern)
    
    patterns_df = pd.DataFrame(patterns)
    logging.debug(f"Erkannte {len(patterns_df)} Candlestick-Formationen")
    return patterns_df

# Funktion zur Erstellung eines Candlestick-Charts
def plot_candlestick_chart(df, patterns, output_file='candlestick_chart.html', max_annotations=100):
    logging.debug(f"Erstelle Candlestick-Chart: {output_file}")
    try:
        fig = make_subplots(rows=1, cols=1)
        
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candlestick'
        ))
        
        if len(patterns) > max_annotations:
            logging.debug(f"Begrenze Annotationen auf {max_annotations} von {len(patterns)}")
            patterns = patterns.tail(max_annotations)
        
        for _, pattern in patterns.iterrows():
            fig.add_vline(x=pattern['timestamp'], line_dash="dash", line_color="red")
            fig.add_annotation(
                x=pattern['timestamp'], y=df['high'].max(),
                text=pattern['pattern'], showarrow=True, arrowhead=1, yshift=10
            )
        
        fig.update_layout(
            title='Candlestick-Chart mit erkannten Formationen (letzte {} Muster)'.format(max_annotations),
            xaxis_title='Datum/Uhrzeit',
            yaxis_title='Preis',
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        output_file = os.path.join('results', output_file)
        os.makedirs('results', exist_ok=True)
        fig.write_html(output_file)
        logging.debug(f"Chart gespeichert: {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Fehler beim Erstellen des Charts {output_file}: {str(e)}")
        return None

# Funktion zur Verarbeitung einer einzelnen Datei
def process_file(file_path, chunk_size=1000):
    logging.info(f"Verarbeite Datei: {file_path}")
    try:
        required_columns = ['open', 'high', 'low', 'close']
        df_info = pd.read_parquet(file_path, engine='pyarrow', columns=required_columns)
        if not all(col in df_info.columns for col in required_columns):
            logging.error(f"Fehlende Spalten in {file_path}: {required_columns}")
            return None, f"Fehlende Spalten in {file_path}: {required_columns}"
        
        num_rows = len(df_info)
        logging.debug(f"Datei {file_path} hat {num_rows} Zeilen")
        all_patterns = []
        
        if num_rows <= chunk_size:
            logging.debug("Datei klein genug, verarbeite ohne Chunks")
            df = df_info
            if isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
                logging.debug("Datetime-Index gefunden, verwendet als timestamp")
            else:
                start_date = pd.to_datetime('2024-08-07 00:00:00')
                df['timestamp'] = [start_date + timedelta(hours=i) for i in range(len(df))]
                logging.debug("Kein datetime-Index, synthetischer timestamp erstellt")
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            patterns = detect_candlestick_patterns(df)
            all_patterns.append(patterns)
        else:
            logging.debug(f"Verarbeite Datei in Chunks von {chunk_size} Zeilen")
            overlap = 2
            chunk_index = 0
            for start in range(0, num_rows, chunk_size - overlap):
                end = min(start + chunk_size, num_rows)
                chunk = df_info.iloc[max(0, start - overlap):end]
                logging.info(f"Verarbeite Chunk {chunk_index} ({len(chunk)} Zeilen)")
                
                if isinstance(chunk.index, pd.DatetimeIndex):
                    chunk['timestamp'] = chunk.index
                    logging.debug(f"Chunk {chunk_index}: Datetime-Index gefunden")
                else:
                    start_date = pd.to_datetime('2024-08-07 00:00:00') + timedelta(hours=start)
                    chunk['timestamp'] = [start_date + timedelta(hours=i) for i in range(len(chunk))]
                    logging.debug(f"Chunk {chunk_index}: Synthetischer timestamp erstellt")
                
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
                patterns = detect_candlestick_patterns(chunk)
                if not patterns.empty:
                    all_patterns.append(patterns)
                chunk_index += 1
        
        if all_patterns:
            patterns_df = pd.concat(all_patterns, ignore_index=True)
            logging.debug(f"Erkannte {len(patterns_df)} Candlestick-Formationen in {file_path}")
        else:
            patterns_df = pd.DataFrame()
            logging.debug(f"Keine Candlestick-Formationen in {file_path} erkannt")
        
        output_file = f"candlestick_chart_{os.path.basename(file_path).replace('.parquet', '')}.html"
        last_chunk = df_info.tail(1000)
        if isinstance(last_chunk.index, pd.DatetimeIndex):
            last_chunk['timestamp'] = last_chunk.index
        else:
            start_date = pd.to_datetime('2024-08-07 00:00:00')
            last_chunk['timestamp'] = [start_date + timedelta(hours=i) for i in range(len(last_chunk))]
        last_chunk['timestamp'] = pd.to_datetime(last_chunk['timestamp'])
        output_file = plot_candlestick_chart(last_chunk, patterns_df, output_file, max_annotations=100)
        if output_file:
            message = f"Grafik gespeichert als: {output_file}"
        else:
            message = f"Fehler beim Speichern des Charts für {file_path}"
        
        return patterns_df, message
    except Exception as e:
        logging.error(f"Fehler bei der Verarbeitung von {file_path}: {str(e)}")
        return None, f"Fehler bei der Verarbeitung von {file_path}: {str(e)}"

# Funktion zur Auswahl von Dateien mit tkinter
def select_files(last_path):
    logging.debug(f"Öffne Dateidialog, Startverzeichnis: {last_path}")
    try:
        root = Tk()
        root.withdraw()
        initial_dir = last_path if os.path.exists(last_path) else os.getcwd()
        file_paths = askopenfilenames(
            title="Wähle Parquet-Dateien aus",
            initialdir=initial_dir,
            filetypes=[("Parquet files", "*.parquet")]
        )
        root.destroy()
        
        if file_paths:
            last_dir = os.path.dirname(file_paths[0])
            with open(os.path.join('results', 'last_path.json'), 'w') as f:
                json.dump({'last_path': last_dir}, f)
            logging.debug(f"Letzter Pfad gespeichert: {last_dir}")
        else:
            logging.warning("Keine Dateien ausgewählt")
        
        return file_paths
    except Exception as e:
        logging.error(f"Fehler im Dateidialog: {str(e)}")
        return []

# Hauptprogramm
def main():
    logging.info("Starte candlestick_analyzer")
    print("Starte candlestick_analyzer...")
    
    last_path = os.getcwd()
    if os.path.exists(os.path.join('results', 'last_path.json')):
        try:
            with open(os.path.join('results', 'last_path.json'), 'r') as f:
                last_path = json.load(f).get('last_path', os.getcwd())
            logging.debug(f"Letzter Pfad geladen: {last_path}")
        except Exception as e:
            logging.error(f"Fehler beim Laden von last_path.json: {str(e)}")
    
    file_paths = select_files(last_path)
    if not file_paths:
        logging.warning("Keine Dateien ausgewählt. Programm wird beendet.")
        print("Keine Dateien ausgewählt. Programm wird beendet.")
        return
    
    all_patterns = []
    for file_path in tqdm(file_paths, desc="Verarbeite Dateien"):
        patterns, message = process_file(file_path, chunk_size=1000)
        print(message)
        logging.info(message)
        if patterns is not None and not patterns.empty:
            all_patterns.append(patterns)
    
    print("\nZusammenfassen der Ergebnisse...")
    logging.info("Zusammenfassen der Ergebnisse")
    if all_patterns:
        try:
            logging.debug(f"Kombiniere {len(all_patterns)} Pattern-DataFrames")
            all_patterns_df = pd.concat(all_patterns, ignore_index=True)
            if all_patterns_df.empty:
                logging.info("Keine Candlestick-Formationen erkannt (leerer DataFrame nach pd.concat)")
                print("Keine Candlestick-Formationen erkannt (leerer DataFrame).")
                return
            
            print("\nErkannte Candlestick-Formationen:")
            print(all_patterns_df.to_string(index=False))
            logging.info("Ergebnisse in Konsole ausgegeben")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs('results', exist_ok=True)
            output_csv = os.path.join('results', f"candlestick_patterns_{timestamp}.csv")
            all_patterns_df.to_csv(output_csv, index=False)
            print(f"Ergebnisse als CSV gespeichert: {output_csv}")
            logging.info(f"Ergebnisse als CSV gespeichert: {output_csv}")
            
            output_excel = os.path.join('results', f"candlestick_patterns_{timestamp}.xlsx")
            all_patterns_df.to_excel(output_excel, index=False, engine='openpyxl')
            print(f"Ergebnisse als Excel gespeichert: {output_excel}")
            logging.info(f"Ergebnisse als Excel gespeichert: {output_excel}")
        except Exception as e:
            logging.error(f"Fehler beim Zusammenfassen oder Speichern der Ergebnisse: {str(e)}")
            print(f"Fehler beim Zusammenfassen oder Speichern der Ergebnisse: {str(e)}")
    else:
        logging.info("Keine Candlestick-Formationen erkannt (keine Pattern-DataFrames)")
        print("Keine Candlestick-Formationen erkannt.")

if __name__ == "__main__":
    main()