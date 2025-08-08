# trading_strategy_kimi_v1.py (FINALE VERSION)
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import ta
import os
import json
import logging
from datetime import datetime, timedelta
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from tqdm import tqdm
from crypto_trading.analysis.candlestick_analyzer import detect_candlestick_patterns
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# === FEHLENDE FUNKTIONEN ===
def select_files(last_path):
    """Dateiauswahl-Funktion"""
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
            os.makedirs('results', exist_ok=True)
            with open(os.path.join('results', 'last_path.json'), 'w') as f:
                json.dump({'last_path': last_dir}, f)
        return file_paths
    except Exception as e:
        logging.error(f"Fehler im Dateidialog: {str(e)}")
        return []

def calculate_metrics(trades_df, equity_df, initial_capital):
    """Basis-Metriken berechnen"""
    metrics = {}
    if trades_df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'final_capital': initial_capital,
            'return_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
    
    metrics['total_trades'] = len(trades_df)
    metrics['win_rate'] = len(trades_df[trades_df['profit'] > 0]) / len(trades_df)
    metrics['total_profit'] = trades_df['profit'].sum()
    metrics['final_capital'] = initial_capital + metrics['total_profit']
    metrics['return_pct'] = (metrics['final_capital'] - initial_capital) / initial_capital * 100
    
    if len(equity_df) > 1:
        equity_df['returns'] = equity_df['capital'].pct_change().fillna(0)
        metrics['sharpe_ratio'] = (equity_df['returns'].mean() * 252) / (equity_df['returns'].std() * np.sqrt(252)) if equity_df['returns'].std() != 0 else 0
        metrics['max_drawdown'] = ((equity_df['capital'].cummax() - equity_df['capital']) / equity_df['capital'].cummax()).max() * 100
    else:
        metrics['sharpe_ratio'] = 0
        metrics['max_drawdown'] = 0
    
    return metrics

# === ERWEITERTE INDIKATOREN ===
def calculate_advanced_indicators(df):
    """Alle Indikatoren inklusive Trend"""
    logging.debug("Berechne erweiterte Indikatoren")
    
    # Simple Moving Averages
    df['sma20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma50'] = ta.trend.sma_indicator(df['close'], window=50)
    
    # Trend-Klassifizierung
    df['trend'] = 'neutral'
    df.loc[df['sma20'] > df['sma50'], 'trend'] = 'bullish'
    df.loc[df['sma20'] < df['sma50'], 'trend'] = 'bearish'
    
    # Weitere Indikatoren
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    
    return df

# === POSITIONS-MANAGEMENT ===
def calculate_position_size(atr, capital, risk_per_trade=0.02):
    """Volatilitätsbasierte Positionsgröße"""
    if pd.isna(atr) or atr == 0:
        return 1000
    risk_amount = capital * risk_per_trade
    return max(100, risk_amount / (2 * atr))

# === HAUPT-STRATEGIE ===
def simulate_enhanced_trades(df, patterns, initial_capital=10000, risk_per_trade=0.02):
    """Erweiterte Trading-Simulation"""
    logging.debug("Starte erweiterte Tradingsimulation")
    
    trades = []
    capital = initial_capital
    positions = []
    equity = []
    
    for i in tqdm(range(len(patterns)), desc="Simuliere erweiterte Trades"):
        if i >= len(df):
            continue
            
        row = patterns.iloc[i]
        price = df.iloc[i]['close']
        timestamp = row['timestamp']
        current_data = df.iloc[i]
        
        # Entry-Bedingungen
        if row['signal'] in ['Kaufen', 'Verkaufen'] and len(positions) < 3:
            is_long = row['signal'] == 'Kaufen'
            
            # Trend-Filter
            trend_ok = (is_long and current_data['trend'] == 'bullish') or \
                      (not is_long and current_data['trend'] == 'bearish')
            
            # ADX-Filter
            strong_trend = current_data['adx'] > 20
            
            if trend_ok and strong_trend:
                position_size = calculate_position_size(
                    current_data['atr'], capital, risk_per_trade
                )
                
                position = {
                    'id': len(trades),
                    'type': 'long' if is_long else 'short',
                    'entry_price': price,
                    'size': position_size,
                    'stop_loss': price - 2 * current_data['atr'] if is_long else price + 2 * current_data['atr'],
                    'take_profit': price + 4 * current_data['atr'] if is_long else price - 4 * current_data['atr'],
                    'timestamp': timestamp,
                    'symbol': df.iloc[0].get('symbol', 'Unknown')
                }
                
                positions.append(position)
                capital -= position_size
        
        # Position-Management
        for pos in positions[:]:
            current_price = df.iloc[i]['close']
            atr = df.iloc[i]['atr']
            
            # Stop-Loss/TP Checks
            stop_hit = (pos['type'] == 'long' and current_price <= pos['stop_loss']) or \
                      (pos['type'] == 'short' and current_price >= pos['stop_loss'])
            
            tp_hit = (pos['type'] == 'long' and current_price >= pos['take_profit']) or \
                    (pos['type'] == 'short' and current_price <= pos['take_profit'])
            
            if stop_hit or tp_hit or (i == len(patterns) - 1):
                profit = (current_price - pos['entry_price']) * pos['size'] if pos['type'] == 'long' else \
                        (pos['entry_price'] - current_price) * pos['size']
                
                capital += pos['size'] + profit
                
                trades.append({
                    'symbol': pos['symbol'],
                    'entry_time': pos['timestamp'],
                    'exit_time': timestamp,
                    'type': 'Kaufen' if pos['type'] == 'long' else 'Verkaufen',
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'size': pos['size'],
                    'profit': profit,
                    'return_pct': (profit / pos['size']) * 100,
                    'duration_hours': (timestamp - pos['timestamp']).total_seconds() / 3600,
                    'exit_reason': 'stop_loss' if stop_hit else ('take_profit' if tp_hit else 'end_of_data')
                })
                
                positions.remove(pos)
        
        # Equity-Tracking
        unrealized = sum([(df.iloc[i]['close'] - p['entry_price']) * p['size'] if p['type'] == 'long' else
                         (p['entry_price'] - df.iloc[i]['close']) * p['size'] 
                         for p in positions])
        equity.append({
            'timestamp': timestamp,
            'capital': capital + unrealized,
            'open_positions': len(positions)
        })
    
    return pd.DataFrame(trades), pd.DataFrame(equity), capital

# === AUSWERTUNG & EXPORT ===
def generate_portfolio_report(all_trades, all_equity, all_metrics, file_paths):
    """Erstellt globale Portfolio-Auswertung"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. GLOBALE ZUSAMMENFASSUNG
    global_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    global_equity = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    
    # 2. GLOBALE METRIKEN
    total_initial_capital = 10000 * len(file_paths)
    global_metrics = calculate_metrics(global_trades, global_equity, total_initial_capital)
    
    # Zusätzliche globale Metriken
    if not global_trades.empty:
        global_metrics.update({
            'total_assets': len(file_paths),
            'best_performing_asset': '',
            'worst_performing_asset': '',
            'total_winning_trades': len(global_trades[global_trades['profit'] > 0]),
            'total_losing_trades': len(global_trades[global_trades['profit'] < 0]),
            'avg_profit_per_trade': global_trades['profit'].mean(),
            'largest_win': global_trades['profit'].max(),
            'largest_loss': global_trades['profit'].min(),
            'profit_factor': abs(global_trades[global_trades['profit'] > 0]['profit'].sum() / 
                               global_trades[global_trades['profit'] < 0]['profit'].sum()),
            'avg_duration': global_trades['duration_hours'].mean()
        })
    
    # 3. DATEI-EXPORT
    os.makedirs('results', exist_ok=True)
    
    # CSV Export
    csv_file = f"results/enhanced_portfolio_trades_{timestamp}.csv"
    global_trades.to_csv(csv_file, index=False)
    
    # Excel Export mit mehreren Sheets
    excel_file = f"results/enhanced_portfolio_analysis_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Einzeltrades
        global_trades.to_excel(writer, sheet_name='Einzeltrades', index=False)
        
        # Monatliche Performance
        if not global_trades.empty:
            monthly_stats = global_trades.groupby(
                global_trades['entry_time'].dt.to_period('M')
            ).agg({
                'profit': ['sum', 'count', 'mean'],
                'return_pct': 'mean'
            })
            monthly_stats.columns = ['Gesamt_Profit', 'Anzahl_Trades', 'Durchschnitt_Profit', 'Durchschnitt_Return']
            monthly_stats.to_excel(writer, sheet_name='Monatsauswertung')
        
        # Symbol-Performance
        if not global_trades.empty:
            symbol_stats = global_trades.groupby('symbol').agg({
                'profit': ['sum', 'count', 'mean', 'std'],
                'return_pct': ['mean', 'std']
            })
            symbol_stats.columns = ['Gesamt_Profit', 'Anzahl_Trades', 'Durchschnitt_Profit', 
                                   'Profit_Std', 'Durchschnitt_Return', 'Return_Std']
            symbol_stats.to_excel(writer, sheet_name='Symbol_Auswertung')
        
        # Globale Metriken
        metrics_df = pd.DataFrame(list(global_metrics.items()), columns=['Metrik', 'Wert'])
        metrics_df.to_excel(writer, sheet_name='Globale_Metriken', index=False)
        
        # Equity-Kurve
        if not global_equity.empty:
            global_equity.to_excel(writer, sheet_name='Equity_Kurve', index=False)
    
    # 4. HTML DASHBOARD
    html_file = f"results/enhanced_portfolio_dashboard_{timestamp}.html"
    create_portfolio_dashboard(global_trades, global_equity, global_metrics, html_file)
    
    return global_metrics, {
        'csv': csv_file,
        'excel': excel_file,
        'html': html_file
    }

def create_portfolio_dashboard(trades_df, equity_df, metrics, output_file):
    """Erstellt interaktives HTML Dashboard"""
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Equity Kurve', 'Monatliche Performance', 
                       'Symbol Verteilung', 'Trade Dauer Verteilung',
                       'Profit Verteilung', 'Drawdown Analyse'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.1
    )
    
    if not trades_df.empty:
        # 1. Equity Kurve
        fig.add_trace(
            go.Scatter(x=equity_df['timestamp'], y=equity_df['capital'],
                      mode='lines', name='Kapital',
                      line=dict(color='green', width=2)),
            row=1, col=1
        )
        
        # 2. Monatliche Performance
        monthly = trades_df.groupby(trades_df['entry_time'].dt.to_period('M'))['profit'].sum()
        fig.add_trace(
            go.Bar(x=monthly.index.astype(str), y=monthly.values,
                   name='Monatlicher Profit', marker_color='blue'),
            row=1, col=2
        )
        
        # 3. Symbol Verteilung
        symbol_profits = trades_df.groupby('symbol')['profit'].sum()
        fig.add_trace(
            go.Bar(x=symbol_profits.index, y=symbol_profits.values,
                   name='Symbol Profit', marker_color='orange'),
            row=2, col=1
        )
        
        # 4. Trade Dauer
        fig.add_trace(
            go.Histogram(x=trades_df['duration_hours'], name='Trade Dauer',
                        nbinsx=50, marker_color='purple'),
            row=2, col=2
        )
        
        # 5. Profit Verteilung
        fig.add_trace(
            go.Histogram(x=trades_df['profit'], name='Profit Verteilung',
                        nbinsx=50, marker_color='red'),
            row=3, col=1
        )
    
    fig.update_layout(
        title=f"🚀 Portfolio Performance Dashboard | Gesamtrendite: {metrics.get('return_pct', 0):.2f}%",
        height=1200,
        showlegend=True
    )
    
    fig.write_html(output_file)
    return output_file

# === HAUPTFUNKTION MIT AUSWERTUNG ===
def main_enhanced():
    logging.info("Starte vollständige Portfolio-Analyse")
    
    # Dateiauswahl
    last_path = os.path.join(os.getcwd(), 'data', 'raw')
    if os.path.exists(os.path.join('results', 'last_path.json')):
        try:
            with open(os.path.join('results', 'last_path.json'), 'r') as f:
                last_path = json.load(f).get('last_path', last_path)
        except:
            pass
    
    file_paths = select_files(last_path)
    if not file_paths:
        print("❌ Keine Dateien ausgewählt")
        return
    
    all_trades = []
    all_equity = []
    all_metrics = []
    
    print(f"📂 Verarbeite {len(file_paths)} Dateien...")
    
    for file_path in tqdm(file_paths, desc="Verarbeite Dateien"):
        try:
            # Daten laden
            df = pd.read_parquet(file_path, engine='pyarrow')
            required_columns = ['open', 'high', 'low', 'close']
            
            if not all(col in df.columns for col in required_columns):
                print(f"⚠️ Überspringe {os.path.basename(file_path)} - fehlende Spalten")
                continue
            
            # Symbol aus Dateiname extrahieren
            symbol = os.path.basename(file_path).split('_')[0]
            df['symbol'] = symbol
            
            # Zeitstempel
            if isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
            else:
                start_date = pd.to_datetime('2024-08-07 00:00:00')
                df['timestamp'] = [start_date + timedelta(hours=i) for i in range(len(df))]
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Indikatoren & Patterns
            df = calculate_advanced_indicators(df)
            patterns = detect_candlestick_patterns(df)
            
            # Simulation
            trades_df, equity_df, final_capital = simulate_enhanced_trades(
                df, patterns, initial_capital=10000, risk_per_trade=0.02
            )
            
            if not trades_df.empty:
                all_trades.append(trades_df)
                all_equity.append(equity_df)
                
                metrics = calculate_metrics(trades_df, equity_df, 10000)
                all_metrics.append({
                    'symbol': symbol,
                    **metrics
                })
                
                print(f"✅ {symbol}: {len(trades_df)} Trades, ${metrics['total_profit']:.2f} Profit")
            
        except Exception as e:
            print(f"❌ Fehler bei {os.path.basename(file_path)}: {str(e)}")
    
    # Globale Auswertung
    if all_trades:
        global_metrics, files = generate_portfolio_report(
            all_trades, all_equity, all_metrics, file_paths
        )
        
        # Konsolen-Ausgabe
        print("\n" + "="*60)
        print("🎯 **GLOBALE PORTFOLIO AUSWERTUNG**")
        print("="*60)
        print(f"📊 Anzahl Assets: {global_metrics['total_assets']}")
        print(f"💰 Gesamtprofit: ${global_metrics['total_profit']:.2f}")
        print(f"📈 Gesamtrendite: {global_metrics['return_pct']:.2f}%")
        print(f"⚖️ Win-Rate: {global_metrics['win_rate']*100:.1f}%")
        print(f"📉 Max Drawdown: {global_metrics['max_drawdown']:.2f}%")
        print(f"🎯 Sharpe Ratio: {global_metrics['sharpe_ratio']:.2f}")
        print(f"⏱️ Durchschn. Trade-Dauer: {global_metrics['avg_duration']:.1f}h")
        
        print("\n📁 **Exportierte Dateien:**")
        print(f"📄 CSV: {files['csv']}")
        print(f"📊 Excel: {files['excel']}")
        print(f"🌐 HTML Dashboard: {files['html']}")
        
    else:
        print("❌ Keine Trades generiert")

if __name__ == "__main__":
    main_enhanced()