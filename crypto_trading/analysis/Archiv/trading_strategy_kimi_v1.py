# trading_strategy.py (erweiterte Version)
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import ta  # pip install ta
import os
import json
import logging
from datetime import datetime, timedelta
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from tqdm import tqdm
from crypto_trading.analysis.candlestick_analyzer import detect_candlestick_patterns
# Füge diese Zeile am Anfang hinzu:
from crypto_trading.analysis.candlestick_analyzer import select_files
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- NEUE HILFSFUNKTIONEN ---
def calculate_advanced_indicators(df):
    """Erweiterte Indikatoren mit ta-Library"""
    logging.debug("Berechne erweiterte Indikatoren")

    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    df['trend'] = 'neutral'
    df.loc[df['sma20'] > df['sma50'], 'trend'] = 'bullish'
    df.loc[df['sma20'] < df['sma50'], 'trend'] = 'bearish'
    
    # Trend-Indikatoren
    df['sma20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['ema12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema26'] = ta.trend.ema_indicator(df['close'], window=26)
    
    # Momentum
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    
    # Volatilität
    df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
    df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close'] * 100
    
    # Volumen
    if 'volume' in df.columns:
        df['vwap'] = ta.volume.volume_weighted_average_price(
            df['high'], df['low'], df['close'], df['volume']
        )
    
    # Trend-Klassifizierung mit ADX
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    df['trend_strength'] = 'weak'
    df.loc[df['adx'] > 25, 'trend_strength'] = 'strong'
    
    # Dynamische ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    
    # === NEU: Trend-Klassifizierung hinzufügen ===
    df['trend'] = 'neutral'
    df.loc[df['sma20'] > df['sma50'], 'trend'] = 'bullish'
    df.loc[df['sma20'] < df['sma50'], 'trend'] = 'bearish'

    return df

def calculate_position_size(atr, capital, risk_per_trade=0.02):
    """Volatilitäts-basierte Positionsgröße"""
    if pd.isna(atr) or atr == 0:
        return 1000  # Fallback
    risk_amount = capital * risk_per_trade
    return max(100, risk_amount / (2 * atr))  # Mindestgröße 100

def trailing_stop_update(current_price, entry_price, stop_loss, is_long, atr):
    """Trailing Stop mit ATR-Abstand"""
    if is_long:
        new_stop = max(stop_loss, current_price - 2 * atr)
    else:
        new_stop = min(stop_loss, current_price + 2 * atr)
    return new_stop

# --- ERWEITERTE TRADE-SIMULATION ---
def simulate_enhanced_trades(df, patterns, initial_capital=10000, risk_per_trade=0.02):
    logging.debug("Starte erweiterte Tradingsimulation")
    
    trades = []
    capital = initial_capital
    positions = []  # Unterstützt mehrere Positionen
    equity = []
    
    for i in tqdm(range(len(patterns)), desc="Simuliere erweiterte Trades"):
        if i >= len(df):
            continue
            
        row = patterns.iloc[i]
        price = df.iloc[i]['close']
        timestamp = row['timestamp']
        
        # Marktbedingungen prüfen
        current_data = df.iloc[i]
        
        # Filtern nach RSI und Trendstärke
        if current_data['rsi'] < 20 and row['signal'] == 'Kaufen':
            signal_strength = 'strong'
        elif current_data['rsi'] > 80 and row['signal'] == 'Verkaufen':
            signal_strength = 'strong'
        else:
            signal_strength = 'normal'
        
        # Entry-Logik
        if row['signal'] in ['Kaufen', 'Verkaufen'] and len(positions) < 3:  # Max 3 Positionen
            is_long = row['signal'] == 'Kaufen'
            
            # Trend-Filter
            trend_ok = (is_long and current_data['trend'] == 'bullish') or \
                      (not is_long and current_data['trend'] == 'bearish')
            
            # ADX-Filter (starke Trends bevorzugen)
            strong_trend = current_data['adx'] > 25
            
            if trend_ok and (strong_trend or signal_strength == 'strong'):
                position_size = calculate_position_size(
                    current_data['atr'], capital, risk_per_trade
                )
                
                entry_price = price
                atr = current_data['atr']
                
                position = {
                    'id': len(trades),
                    'type': 'long' if is_long else 'short',
                    'entry_price': entry_price,
                    'size': position_size,
                    'stop_loss': entry_price - 2 * atr if is_long else entry_price + 2 * atr,
                    'take_profit': entry_price + 4 * atr if is_long else entry_price - 4 * atr,
                    'timestamp': timestamp,
                    'trail_active': False
                }
                
                positions.append(position)
                capital -= position_size  # Margin-Reservierung
                logging.debug(f"Position eröffnet: {position}")
        
        # Position-Management
        for pos in positions[:]:  # Kopie für sicheres Iterieren
            current_price = df.iloc[i]['close']
            atr = df.iloc[i]['atr']
            
            # Trailing Stop aktivieren nach 1 ATR Gewinn
            unrealized_pnl = (current_price - pos['entry_price']) * pos['size'] if pos['type'] == 'long' else \
                            (pos['entry_price'] - current_price) * pos['size']
            
            if unrealized_pnl > atr * pos['size'] and not pos['trail_active']:
                pos['trail_active'] = True
                logging.debug(f"Trailing Stop aktiviert für Position {pos['id']}")
            
            # Stop-Loss/TP Checks
            stop_hit = (pos['type'] == 'long' and current_price <= pos['stop_loss']) or \
                      (pos['type'] == 'short' and current_price >= pos['stop_loss'])
            
            tp_hit = (pos['type'] == 'long' and current_price >= pos['take_profit']) or \
                    (pos['type'] == 'short' and current_price <= pos['take_profit'])
            
            # Trailing Stop Update
            if pos['trail_active']:
                pos['stop_loss'] = trailing_stop_update(
                    current_price, pos['entry_price'], pos['stop_loss'], 
                    pos['type'] == 'long', atr
                )
            
            if stop_hit or tp_hit or (i == len(patterns) - 1):  # Close at end
                exit_price = current_price
                profit = (exit_price - pos['entry_price']) * pos['size'] if pos['type'] == 'long' else \
                        (pos['entry_price'] - exit_price) * pos['size']
                
                capital += pos['size'] + profit
                
                trades.append({
                    'entry_time': pos['timestamp'],
                    'exit_time': timestamp,
                    'type': 'Kaufen' if pos['type'] == 'long' else 'Verkaufen',
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'size': pos['size'],
                    'profit': profit,
                    'return_pct': (profit / pos['size']) * 100,
                    'duration_hours': (timestamp - pos['timestamp']).total_seconds() / 3600,
                    'exit_reason': 'stop_loss' if stop_hit else ('take_profit' if tp_hit else 'end_of_data')
                })
                
                positions.remove(pos)
                logging.debug(f"Position geschlossen: {profit}")
        
        # Equity-Tracking
        unrealized = sum([(df.iloc[i]['close'] - p['entry_price']) * p['size'] if p['type'] == 'long' else
                         (p['entry_price'] - df.iloc[i]['close']) * p['size'] 
                         for p in positions])
        equity.append({
            'timestamp': timestamp,
            'capital': capital + unrealized,
            'open_positions': len(positions)
        })
    
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity)
    return trades_df, equity_df, capital

# --- ERWEITERTE METRIKEN ---
def calculate_enhanced_metrics(trades_df, equity_df, initial_capital):
    metrics = calculate_metrics(trades_df, equity_df, initial_capital)
    
    if not trades_df.empty:
        # Risk-Adjusted Returns
        metrics['calmar_ratio'] = metrics['return_pct'] / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else 0
        
        # Trade-Statistiken
        metrics['avg_duration_hours'] = trades_df['duration_hours'].mean()
        metrics['largest_win'] = trades_df['profit'].max()
        metrics['largest_loss'] = trades_df['profit'].min()
        metrics['profit_factor'] = trades_df[trades_df['profit'] > 0]['profit'].sum() / \
                                  abs(trades_df[trades_df['profit'] < 0]['profit'].sum()) \
                                  if trades_df['profit'].min() < 0 else float('inf')
        
        # Performance nach Exit-Grund
        exit_stats = trades_df.groupby('exit_reason')['profit'].agg(['sum', 'count'])
        metrics['exit_stats'] = exit_stats.to_dict()
    
    return metrics

# --- ANGEPASSTE HAUPTFUNKTIONEN ---
def process_enhanced_file(file_path, initial_capital=10000, risk_per_trade=0.02):
    logging.info(f"Verarbeite Datei mit erweiterter Strategie: {file_path}")
    
    try:
        required_columns = ['open', 'high', 'low', 'close']
        df = pd.read_parquet(file_path, engine='pyarrow')
        
        if not all(col in df.columns for col in required_columns):
            return None, None, None, f"Fehlende Spalten: {required_columns}"
        
        # Erweiterte Indikatoren
        df = calculate_advanced_indicators(df)
        
        # Zeitstempel
        if isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = df.index
        else:
            start_date = pd.to_datetime('2024-08-07 00:00:00')
            df['timestamp'] = [start_date + timedelta(hours=i) for i in range(len(df))]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Candlestick-Patterns
        patterns = detect_candlestick_patterns(df)
        
        # Erweiterte Simulation
        trades_df, equity_df, final_capital = simulate_enhanced_trades(
            df, patterns, initial_capital, risk_per_trade
        )
        
        # Erweiterte Metriken
        metrics = calculate_enhanced_metrics(trades_df, equity_df, initial_capital)
        
        # Visualisierung
        output_file = f"enhanced_trading_{os.path.basename(file_path).replace('.parquet', '')}.html"
        output_file = plot_trading_results(df.tail(1000), trades_df, equity_df, output_file)

        # === NEU: Überprüfung der Spalten ===
        required_cols = ['trend', 'rsi', 'adx', 'atr']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, None, None, f"Fehlende Spalten: {missing_cols}"
        
        return trades_df, equity_df, metrics, f"Ergebnisse gespeichert: {output_file}"
        
    except Exception as e:
        logging.error(f"Fehler: {str(e)}")
        return None, None, None, f"Fehler: {str(e)}"

# --- ABWÄRTSKOMPATIBEL ---
# Neue main() Funktion für erweiterte Strategie
def main_enhanced():
    logging.info("Starte erweiterte Trading-Strategie")
    
    # Dateiauswahl (gleich wie vorher)
    last_path = os.path.join(os.getcwd(), 'data', 'raw')
    if os.path.exists(os.path.join('results', 'last_path.json')):
        try:
            with open(os.path.join('results', 'last_path.json'), 'r') as f:
                last_path = json.load(f).get('last_path', last_path)
        except:
            pass
    
    file_paths = select_files(last_path)
    if not file_paths:
        return
    
    all_trades = []
    all_equity = []
    all_metrics = []
    
    for file_path in tqdm(file_paths, desc="Verarbeite Dateien"):
        trades_df, equity_df, metrics, message = process_enhanced_file(file_path)
        print(message)

        # === NEU: Sofortige Auswertung ===
        if trades_df is not None and not trades_df.empty:
            print(f"\n📈 **{os.path.basename(file_path)}**")
            print(f"   📊 Trades: {len(trades_df)}")
            print(f"   💰 Gesamt-Profit: ${trades_df['profit'].sum():.2f}")
            print(f"   📈 Win-Rate: {(trades_df['profit'] > 0).mean()*100:.1f}%")
            print(f"   ⏱️ Durchschn. Dauer: {trades_df['duration_hours'].mean():.1f}h")
            
            # Top 3 Gewinner/Verlierer
            top_wins = trades_df.nlargest(3, 'profit')[['entry_time', 'profit', 'type']]
            top_losses = trades_df.nsmallest(3, 'profit')[['entry_time', 'profit', 'type']]
            
            print("\n🏆 **Top Gewinner:**")
            print(top_wins.to_string(index=False))
            
            print("\n💸 **Top Verlierer:**")
            print(top_losses.to_string(index=False))
        
        if trades_df is not None:
            all_trades.append(trades_df)
            all_equity.append(equity_df)
            all_metrics.append(metrics)
    
    if all_trades:
        # Zusammenfassung
        combined_trades = pd.concat(all_trades, ignore_index=True)
        combined_equity = pd.concat(all_equity, ignore_index=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV Export
        combined_trades.to_csv(f"results/enhanced_trades_{timestamp}.csv", index=False)
        
        # Excel mit mehreren Sheets
        with pd.ExcelWriter(f"results/enhanced_analysis_{timestamp}.xlsx", engine='openpyxl') as writer:
            combined_trades.to_excel(writer, sheet_name='Trades', index=False)
            combined_equity.to_excel(writer, sheet_name='Equity', index=False)
            
            # Zusätzliche Analysen
            monthly_stats = combined_trades.groupby(
                combined_trades['entry_time'].dt.to_period('M')
            )['profit'].sum()
            monthly_stats.to_excel(writer, sheet_name='Monthly_PnL')
        
        print("\n=== ERWEITERTE METRIKEN ===")
        for i, metrics in enumerate(all_metrics):
            print(f"\nDatei {os.path.basename(file_paths[i])}:")
            for key, value in metrics.items():
                if key != 'exit_stats':
                    print(f"{key}: {value:.2f}")

    

if __name__ == "__main__":
    main_enhanced()