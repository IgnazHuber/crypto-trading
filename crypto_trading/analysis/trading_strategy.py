import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import logging
from datetime import datetime
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from tqdm import tqdm
from crypto_trading.analysis.candlestick_analyzer import detect_candlestick_patterns
import warnings

# Unterdrücke Plotly-Warnungen
warnings.filterwarnings("ignore", category=UserWarning)

# Logging einrichten
try:
    os.makedirs('results', exist_ok=True)
    logging.basicConfig(
        filename=os.path.join('results', 'trading_strategy.log'),
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
except Exception as e:
    print(f"Fehler beim Einrichten des Loggings: {str(e)}")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Funktion zur Berechnung von Indikatoren (Marktumfeld)
def calculate_indicators(df):
    logging.debug("Berechne Marktumfeld-Indikatoren")
    for _ in tqdm(range(1), desc="Berechne Indikatoren", leave=False):
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        df['trend'] = 'neutral'
        df.loc[df['sma20'] > df['sma50'], 'trend'] = 'bullish'
        df.loc[df['sma20'] < df['sma50'], 'trend'] = 'bearish'
        
        # ATR (Average True Range)
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
    
    return df

# Funktion zur Simulation von Trades
def simulate_trades(df, patterns, initial_capital=10000, position_size=1000):
    logging.debug("Starte Tradingsimulation")
    trades = []
    capital = initial_capital
    position = None
    equity = []
    
    for i in tqdm(range(len(patterns)), desc="Simuliere Trades", leave=False):
        if i >= len(df):
            continue
        
        row = patterns.iloc[i]
        price = df.iloc[i]['close']
        timestamp = row['timestamp']
        atr = df.iloc[i]['atr']
        
        if row['signal'] == 'Kaufen' and df.iloc[i]['trend'] == 'bullish' and not position:
            entry_price = price
            stop_loss = entry_price - 2 * atr
            take_profit = entry_price + 4 * atr
            position = {'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit, 'timestamp': timestamp}
            logging.debug(f"Kauf-Trade eröffnet: {timestamp}, Preis: {entry_price}")
        
        elif row['signal'] == 'Verkaufen' and df.iloc[i]['trend'] == 'bearish' and not position:
            entry_price = price
            stop_loss = entry_price + 2 * atr
            take_profit = entry_price - 4 * atr
            position = {'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit, 'timestamp': timestamp}
            logging.debug(f"Verkauf-Trade eröffnet: {timestamp}, Preis: {entry_price}")
        
        if position:
            current_price = df.iloc[i]['close']
            if position['stop_loss'] >= current_price or position['take_profit'] <= current_price:
                exit_price = current_price
                if row['signal'] == 'Kaufen':
                    profit = (exit_price - position['entry_price']) * (position_size / position['entry_price'])
                else:
                    profit = (position['entry_price'] - exit_price) * (position_size / position['entry_price'])
                capital += profit
                trades.append({
                    'entry_time': position['timestamp'],
                    'exit_time': timestamp,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': profit,
                    'type': row['signal']
                })
                logging.debug(f"Trade geschlossen: {timestamp}, Gewinn/Verlust: {profit}")
                position = None
            equity.append({'timestamp': timestamp, 'capital': capital})
    
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity)
    logging.debug(f"Simulation abgeschlossen: {len(trades)} Trades, Endkapital: {capital}")
    return trades_df, equity_df, capital

# Funktion zur Berechnung von Metriken
def calculate_metrics(trades_df, equity_df, initial_capital):
    logging.debug("Berechne Metriken")
    metrics = {}
    metrics['total_trades'] = len(trades_df)
    metrics['win_rate'] = len(trades_df[trades_df['profit'] > 0]) / len(trades_df) if len(trades_df) > 0 else 0
    metrics['total_profit'] = trades_df['profit'].sum()
    metrics['final_capital'] = initial_capital + metrics['total_profit']
    metrics['return_pct'] = (metrics['final_capital'] - initial_capital) / initial_capital * 100
    
    equity_df['returns'] = equity_df['capital'].pct_change().fillna(0)
    metrics['sharpe_ratio'] = (equity_df['returns'].mean() * 252) / (equity_df['returns'].std() * (252 ** 0.5)) if equity_df['returns'].std() != 0 else 0
    metrics['max_drawdown'] = ((equity_df['capital'].cummax() - equity_df['capital']) / equity_df['capital'].cummax()).max() * 100
    
    return metrics

# Funktion zur Visualisierung der Ergebnisse
def plot_trading_results(df, trades_df, equity_df, output_file='trading_results.html'):
    logging.debug(f"Erstelle Trading-Chart: {output_file}")
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Candlestick-Chart mit Trades", "Equity-Kurve"))
        
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candlestick'
        ), row=1, col=1)
        
        buys = trades_df[trades_df['type'] == 'Kaufen']
        sells = trades_df[trades_df['type'] == 'Verkaufen']
        fig.add_trace(go.Scatter(
            x=buys['entry_time'], y=buys['entry_price'], mode='markers', name='Kauf',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sells['entry_time'], y=sells['entry_price'], mode='markers', name='Verkauf',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=equity_df['timestamp'], y=equity_df['capital'], mode='lines', name='Equity',
            line=dict(color='blue')
        ), row=2, col=1)
        
        fig.update_layout(
            title='Trading-Strategie Ergebnisse',
            xaxis2_title='Datum/Uhrzeit',
            yaxis_title='Preis',
            yaxis2_title='Kapital',
            xaxis_rangeslider_visible=False,
            height=1000
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
def process_file(file_path, chunk_size=1000, initial_capital=10000, position_size=1000):
    logging.info(f"Verarbeite Datei: {file_path}")
    print(f"Verarbeite Datei: {os.path.basename(file_path)}")
    try:
        required_columns = ['open', 'high', 'low', 'close']
        df = pd.read_parquet(file_path, engine='pyarrow')
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Fehlende Spalten in {file_path}: {required_columns}")
            return None, None, None, f"Fehlende Spalten in {file_path}: {required_columns}"
        
        num_rows = len(df)
        logging.debug(f"Datei {file_path} hat {num_rows} Zeilen")
        print(f"Anzahl Zeilen: {num_rows}")
        
        if isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = df.index
        else:
            start_date = pd.to_datetime('2024-08-07 00:00:00')
            df['timestamp'] = [start_date + timedelta(hours=i) for i in range(len(df))]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Indikatoren berechnen
        df = calculate_indicators(df)
        print("Indikatoren berechnet")
        
        # Candlestick-Formationen erkennen
        patterns = detect_candlestick_patterns(df)
        print(f"Erkannte {len(patterns)} Candlestick-Formationen")
        
        # Trades simulieren
        trades_df, equity_df, final_capital = simulate_trades(df, patterns, initial_capital, position_size)
        print(f"Trades simuliert: {len(trades_df)} Trades")
        
        # Metriken berechnen
        metrics = calculate_metrics(trades_df, equity_df, initial_capital)
        print("Metriken berechnet")
        
        # Ergebnisse visualisieren
        output_file = f"trading_results_{os.path.basename(file_path).replace('.parquet', '')}.html"
        output_file = plot_trading_results(df.tail(1000), trades_df, equity_df, output_file)
        if output_file:
            message = f"Trading-Chart gespeichert als: {output_file}"
        else:
            message = f"Fehler beim Speichern des Trading-Charts für {file_path}"
        
        return trades_df, equity_df, metrics, message
    except Exception as e:
        logging.error(f"Fehler bei der Verarbeitung von {file_path}: {str(e)}")
        return None, None, None, f"Fehler bei der Verarbeitung von {file_path}: {str(e)}"

# Funktion zur Auswahl von Dateien mit tkinter
def select_files(last_path):
    logging.debug(f"Öffne Dateidialog, Startverzeichnis: {last_path}")
    print(f"Öffne Dateidialog, Startverzeichnis: {last_path}")
    try:
        root = Tk()
        root.withdraw()
        initial_dir = last_path if os.path.exists(last_path) else os.path.join(os.getcwd(), 'data', 'raw')
        file_paths = askopenfilenames(
            title="Wähle Parquet-Dateien aus",
            initialdir=initial_dir,
            filetypes=[("Parquet files", "*.parquet")]
        )
        root.destroy()
        
        if file_paths:
            last_dir = os.path.dirname(file_paths[0])
            os.makedirs('results', exist_ok=True)
            with open(os.path.join('results', 'last_path.json'), 'w') as f:
                json.dump({'last_path': last_dir}, f)
            logging.debug(f"Letzter Pfad gespeichert: {last_dir}")
            print(f"Letzter Pfad gespeichert: {last_dir}")
        else:
            logging.warning("Keine Dateien ausgewählt")
            print("Keine Dateien ausgewählt")
        
        return file_paths
    except Exception as e:
        logging.error(f"Fehler im Dateidialog: {str(e)}")
        print(f"Fehler im Dateidialog: {str(e)}")
        return []

# Hauptprogramm
def main():
    logging.info("Starte trading_strategy")
    print("Starte trading_strategy...")
    
    last_path = os.path.join(os.getcwd(), 'data', 'raw')
    if os.path.exists(os.path.join('results', 'last_path.json')):
        try:
            with open(os.path.join('results', 'last_path.json'), 'r') as f:
                last_path = json.load(f).get('last_path', last_path)
            logging.debug(f"Letzter Pfad geladen: {last_path}")
            print(f"Letzter Pfad geladen: {last_path}")
        except Exception as e:
            logging.error(f"Fehler beim Laden von last_path.json: {str(e)}")
            print(f"Fehler beim Laden von last_path.json: {str(e)}")
    
    file_paths = select_files(last_path)
    if not file_paths:
        logging.warning("Keine Dateien ausgewählt. Programm wird beendet.")
        print("Keine Dateien ausgewählt. Programm wird beendet.")
        return
    
    all_trades = []
    all_equity = []
    all_metrics = []
    
    for file_path in tqdm(file_paths, desc="Verarbeite Dateien"):
        trades_df, equity_df, metrics, message = process_file(file_path)
        print(message)
        logging.info(message)
        if trades_df is not None and not trades_df.empty:
            all_trades.append(trades_df)
            all_equity.append(equity_df)
            all_metrics.append(metrics)
    
    print("\nZusammenfassen der Ergebnisse...")
    logging.info("Zusammenfassen der Ergebnisse")
    if all_trades:
        try:
            all_trades_df = pd.concat(all_trades, ignore_index=True)
            all_equity_df = pd.concat(all_equity, ignore_index=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs('results', exist_ok=True)
            
            output_csv = os.path.join('results', f"trading_trades_{timestamp}.csv")
            all_trades_df.to_csv(output_csv, index=False)
            print(f"Trades als CSV gespeichert: {output_csv}")
            logging.info(f"Trades als CSV gespeichert: {output_csv}")
            
            output_excel = os.path.join('results', f"trading_trades_{timestamp}.xlsx")
            all_trades_df.to_excel(output_excel, index=False, engine='openpyxl')
            print(f"Trades als Excel gespeichert: {output_excel}")
            logging.info(f"Trades als Excel gespeichert: {output_excel}")
            
            print("\nTrading-Metriken:")
            for i, metrics in enumerate(all_metrics):
                print(f"\nDatei {os.path.basename(file_paths[i])}:")
                for key, value in metrics.items():
                    print(f"{key}: {value}")
            logging.info("Metriken ausgegeben")
        except Exception as e:
            logging.error(f"Fehler beim Zusammenfassen oder Speichern der Ergebnisse: {str(e)}")
            print(f"Fehler beim Zusammenfassen oder Speichern der Ergebnisse: {str(e)}")
    else:
        logging.info("Keine Trades durchgeführt")
        print("Keine Trades durchgeführt.")

if __name__ == "__main__":
    main()