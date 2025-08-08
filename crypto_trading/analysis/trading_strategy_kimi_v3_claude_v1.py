# trading_strategy_kimi_v4.py – KORRIGIERT & OPTIMIERT
# Vollständige Korrektur mit allen Features + Verbesserungen
# • Robuste Error-Handling
# • Performance-Optimierungen
# • Vollständige Equity-Berechnung
# • Erweiterte Metriken und Visualisierungen

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
import warnings
from typing import Dict, List, Tuple, Optional
import traceback

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# === LOGGING SETUP ===
def setup_logging():
    """Erweiterte Logging-Konfiguration"""
    os.makedirs('results', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('results', 'trading_strategy.log')),
            logging.StreamHandler()
        ]
    )

# === BASIS-FUNKTIONEN ===
def select_files(last_path: str) -> List[str]:
    """Dateiauswahl mit verbessertem Error-Handling"""
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
            try:
                with open(os.path.join('results', 'last_path.json'), 'w') as f:
                    json.dump({'last_path': last_dir}, f)
            except Exception as e:
                logging.warning(f"Konnte letzten Pfad nicht speichern: {e}")
        
        return list(file_paths)
    except Exception as e:
        logging.error(f"Fehler im Dateidialog: {str(e)}")
        return []

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Sichere Division mit Fallback"""
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default

def calculate_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame, initial_capital: float) -> Dict:
    """Basis-Metriken mit robustem Error-Handling"""
    try:
        if trades_df.empty:
            return {
                'total_trades': 0, 'win_rate': 0.0, 'total_profit': 0.0,
                'final_capital': initial_capital, 'return_pct': 0.0,
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0
            }
        
        metrics = {}
        metrics['total_trades'] = len(trades_df)
        
        # Win-Rate berechnen
        winning_trades = trades_df[trades_df['profit'] > 0]
        metrics['win_rate'] = safe_divide(len(winning_trades), len(trades_df), 0.0)
        
        # Profit-Metriken
        metrics['total_profit'] = trades_df['profit'].sum()
        metrics['final_capital'] = initial_capital + metrics['total_profit']
        metrics['return_pct'] = safe_divide(
            metrics['final_capital'] - initial_capital, 
            initial_capital, 0.0
        ) * 100
        
        # Erweiterte Metriken
        if len(equity_df) > 1:
            equity_df = equity_df.copy()
            equity_df['returns'] = equity_df['capital'].pct_change().fillna(0)
            
            # Sharpe Ratio
            mean_return = equity_df['returns'].mean()
            std_return = equity_df['returns'].std()
            metrics['sharpe_ratio'] = safe_divide(
                mean_return * 252, std_return * np.sqrt(252), 0.0
            ) if std_return > 0 else 0.0
            
            # Maximum Drawdown
            equity_series = equity_df['capital']
            running_max = equity_series.expanding().max()
            drawdown = (running_max - equity_series) / running_max
            metrics['max_drawdown'] = drawdown.max() * 100 if not drawdown.empty else 0.0
        else:
            metrics['sharpe_ratio'] = 0.0
            metrics['max_drawdown'] = 0.0
        
        return metrics
    except Exception as e:
        logging.error(f"Fehler bei Metrik-Berechnung: {e}")
        return {
            'total_trades': 0, 'win_rate': 0.0, 'total_profit': 0.0,
            'final_capital': initial_capital, 'return_pct': 0.0,
            'sharpe_ratio': 0.0, 'max_drawdown': 0.0
        }

# === ERWEITERTE INDIKATOREN ===
def calculate_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnung aller technischen Indikatoren mit Error-Handling"""
    try:
        df = df.copy()
        
        # Moving Averages
        df['sma20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # Trend-Bestimmung
        df['trend'] = 'neutral'
        df.loc[df['sma20'] > df['sma50'], 'trend'] = 'bullish'
        df.loc[df['sma20'] < df['sma50'], 'trend'] = 'bearish'
        
        # Momentum-Indikatoren
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        
        # Volatilität
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
        
        # Bollinger Band Width mit sicherer Division
        df['bb_width'] = np.where(
            df['close'] != 0,
            (df['bb_upper'] - df['bb_lower']) / df['close'] * 100,
            0
        )
        
        # ADX für Trend-Stärke
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        
        # Market Regime
        df['regime'] = 'range'
        df.loc[df['adx'] > 25, 'regime'] = 'trending'
        
        # Volatilitäts-Regime
        bb_width_ma = df['bb_width'].rolling(20).mean()
        df.loc[df['bb_width'] < bb_width_ma, 'regime'] = 'low_vol'
        
        # Zeit-Features
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['weekday'] = df.index.weekday
        else:
            df['hour'] = 0
            df['weekday'] = 0
        
        # NaN-Werte durch Forward-Fill ersetzen
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        
        return df
    except Exception as e:
        logging.error(f"Fehler bei Indikator-Berechnung: {e}")
        # Fallback: Mindest-Indikatoren
        df['sma20'] = df['close'].rolling(20).mean()
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        df['trend'] = 'neutral'
        df['regime'] = 'range'
        df['hour'] = 0
        df['weekday'] = 0
        return df

# === RISIKO / POSITION / FILTER ===
def calculate_position_size(capital: float, max_position_pct: float = 0.10, 
                          max_risk_pct: float = 0.03, atr: Optional[float] = None) -> float:
    """Verbesserte Positionsgrößen-Berechnung"""
    try:
        max_position = capital * max_position_pct
        max_risk_amount = capital * max_risk_pct
        
        if atr and atr > 0:
            volatility_adjusted = max_risk_amount / (atr * 1.5)
            return max(min(max_position, volatility_adjusted), 100)
        
        return max(max_position, 100)
    except Exception as e:
        logging.warning(f"Fehler bei Positionsgrößen-Berechnung: {e}")
        return capital * 0.05  # Fallback: 5% des Kapitals

def calculate_risk_levels(entry_price: float, max_loss_pct: float = 0.03, 
                         max_gain_pct: float = 0.20) -> Tuple[float, float]:
    """Stop-Loss und Take-Profit Berechnung"""
    try:
        stop_loss = entry_price * (1 - max_loss_pct)
        take_profit = entry_price * (1 + max_gain_pct)
        return stop_loss, take_profit
    except Exception:
        return entry_price * 0.97, entry_price * 1.20

def volatility_filter(current_data: pd.Series) -> bool:
    """Volatilitäts-Filter"""
    try:
        vol_ratio = safe_divide(current_data.get('atr', 0), current_data.get('close', 1))
        return 0.005 < vol_ratio < 0.05
    except Exception:
        return True

def time_filter(timestamp) -> bool:
    """Zeit-Filter"""
    try:
        hour = timestamp.hour if hasattr(timestamp, 'hour') else 12
        return 4 <= hour <= 22
    except Exception:
        return True

def regime_filter(current_data: pd.Series) -> bool:
    """Market-Regime Filter"""
    try:
        regime = current_data.get('regime', 'range')
        return regime in ['trending', 'range']
    except Exception:
        return True

# === HAUPT-STRATEGIE ===
def simulate_advanced_strategy(df: pd.DataFrame, patterns: pd.DataFrame, 
                             initial_capital: float = 10000,
                             max_position_pct: float = 0.10, 
                             max_risk_pct: float = 0.03) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Erweiterte Trading-Strategie mit robustem Error-Handling"""
    
    logging.info(f"Starte Strategie-Simulation für {df.iloc[0].get('symbol', 'Unknown')}")
    
    trades = []
    capital = initial_capital
    positions = []
    equity = []
    symbol = df.iloc[0].get('symbol', 'Unknown')
    
    try:
        # Sicherstellen, dass DataFrame-Längen übereinstimmen
        min_length = min(len(df), len(patterns))
        if min_length == 0:
            logging.warning("Leere DataFrames - keine Simulation möglich")
            return pd.DataFrame(), pd.DataFrame(), initial_capital
        
        # Progress-Tracking
        for i in tqdm(range(min_length), desc=f"Simuliere {symbol}", leave=False):
            try:
                row = patterns.iloc[i]
                current_price = df.iloc[i]['close']
                timestamp = row.get('timestamp', df.iloc[i].get('timestamp'))
                current_data = df.iloc[i]
                
                # Entry-Logik
                if (row.get('signal') in ['Kaufen', 'Verkaufen'] and
                    len(positions) < 3 and
                    time_filter(timestamp) and
                    volatility_filter(current_data) and
                    regime_filter(current_data)):
                    
                    is_long = row.get('signal') == 'Kaufen'
                    stop_loss, take_profit = calculate_risk_levels(current_price)
                    position_size = calculate_position_size(
                        capital, max_position_pct, max_risk_pct, current_data.get('atr')
                    )
                    
                    position = {
                        'type': 'long' if is_long else 'short',
                        'entry_price': current_price,
                        'size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'max_loss': position_size * 0.03,
                        'max_gain': position_size * 0.20,
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'pyramid_levels': 0,
                        'regime': current_data.get('regime', 'range'),
                        'hour': getattr(timestamp, 'hour', 0),
                        'weekday': getattr(timestamp, 'weekday', 0)
                    }
                    positions.append(position)
                
                # Position-Management
                positions_to_remove = []
                for pos_idx, pos in enumerate(positions):
                    try:
                        # P&L Berechnung
                        if pos['type'] == 'long':
                            current_pnl = (current_price - pos['entry_price']) * pos['size']
                        else:
                            current_pnl = (pos['entry_price'] - current_price) * pos['size']
                        
                        exit_price = current_price
                        exit_reason = None
                        profit = 0
                        
                        # Exit-Bedingungen
                        if pos['type'] == 'long':
                            if current_price <= pos['stop_loss']:
                                exit_reason = 'stop_loss'
                                exit_price = pos['stop_loss']
                                profit = (exit_price - pos['entry_price']) * pos['size']
                            elif current_price >= pos['take_profit']:
                                exit_reason = 'take_profit'
                                exit_price = pos['take_profit']
                                profit = (exit_price - pos['entry_price']) * pos['size']
                        else:  # short
                            if current_price >= pos['stop_loss']:
                                exit_reason = 'stop_loss'
                                exit_price = pos['stop_loss']
                                profit = (pos['entry_price'] - exit_price) * pos['size']
                            elif current_price <= pos['take_profit']:
                                exit_reason = 'take_profit'
                                exit_price = pos['take_profit']
                                profit = (pos['entry_price'] - exit_price) * pos['size']
                        
                        # End-of-Data Exit
                        if not exit_reason and (i == min_length - 1):
                            exit_reason = 'end_of_data'
                            if pos['type'] == 'long':
                                profit = (current_price - pos['entry_price']) * pos['size']
                            else:
                                profit = (pos['entry_price'] - current_price) * pos['size']
                        
                        # Trade schließen
                        if exit_reason:
                            capital += profit
                            
                            # Duration berechnen
                            duration_hours = 0
                            try:
                                if hasattr(timestamp, 'total_seconds') and hasattr(pos['timestamp'], 'total_seconds'):
                                    duration_hours = (timestamp - pos['timestamp']).total_seconds() / 3600
                                elif isinstance(timestamp, (pd.Timestamp, datetime)) and isinstance(pos['timestamp'], (pd.Timestamp, datetime)):
                                    duration_hours = (timestamp - pos['timestamp']).total_seconds() / 3600
                            except Exception:
                                duration_hours = 1.0  # Fallback
                            
                            trade_record = {
                                'symbol': pos['symbol'],
                                'entry_time': pos['timestamp'],
                                'exit_time': timestamp,
                                'type': 'Kaufen' if pos['type'] == 'long' else 'Verkaufen',
                                'entry_price': pos['entry_price'],
                                'exit_price': exit_price,
                                'size': pos['size'],
                                'profit': profit,
                                'return_pct': safe_divide(profit, pos['size'], 0.0) * 100,
                                'capital_before': capital - profit,
                                'capital_after': capital,
                                'duration_hours': max(duration_hours, 0.1),
                                'exit_reason': exit_reason,
                                'max_loss': pos['max_loss'],
                                'max_gain': pos['max_gain'],
                                'risk_reward_ratio': safe_divide(pos['max_gain'], pos['max_loss'], 1.0),
                                'regime': pos['regime'],
                                'pyramid_levels': pos['pyramid_levels'],
                                'hour': pos['hour'],
                                'weekday': pos['weekday']
                            }
                            trades.append(trade_record)
                            positions_to_remove.append(pos_idx)
                    
                    except Exception as e:
                        logging.warning(f"Fehler beim Position-Management: {e}")
                        continue
                
                # Positionen entfernen (rückwärts um Index-Probleme zu vermeiden)
                for idx in reversed(positions_to_remove):
                    positions.pop(idx)
                
                # Equity-Tracking
                unrealized_pnl = 0
                for pos in positions:
                    try:
                        if pos['type'] == 'long':
                            unrealized_pnl += (current_price - pos['entry_price']) * pos['size']
                        else:
                            unrealized_pnl += (pos['entry_price'] - current_price) * pos['size']
                    except Exception:
                        continue
                
                equity.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'capital': capital + unrealized_pnl,  # Total Equity inkl. unrealisierte Gewinne
                    'realized_capital': capital,  # Nur realisierte Gewinne
                    'open_positions': len(positions),
                    'unrealized_pnl': unrealized_pnl
                })
                
            except Exception as e:
                logging.warning(f"Fehler in Iteration {i}: {e}")
                continue
        
        # Abschließende Bereinigung
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity)
        
        logging.info(f"Simulation abgeschlossen: {len(trades_df)} Trades, Endkapital: {capital:,.2f}€")
        
        return trades_df, equity_df, capital
        
    except Exception as e:
        logging.error(f"Kritischer Fehler in Strategie-Simulation: {e}")
        logging.error(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame(), initial_capital

# === ERWEITERTE METRIKEN ===
def calculate_enhanced_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame, 
                             initial_capital: float) -> Dict:
    """Umfassende Metriken-Berechnung"""
    try:
        metrics = calculate_metrics(trades_df, equity_df, initial_capital)
        
        if not trades_df.empty:
            # Erweiterte Trade-Statistiken
            winning_trades = trades_df[trades_df['profit'] > 0]
            losing_trades = trades_df[trades_df['profit'] < 0]
            
            metrics.update({
                'total_winning_trades': len(winning_trades),
                'total_losing_trades': len(losing_trades),
                'avg_profit_per_trade': trades_df['profit'].mean(),
                'avg_win': winning_trades['profit'].mean() if not winning_trades.empty else 0.0,
                'avg_loss': losing_trades['profit'].mean() if not losing_trades.empty else 0.0,
                'largest_win': trades_df['profit'].max(),
                'largest_loss': trades_df['profit'].min(),
                'avg_duration': trades_df['duration_hours'].mean(),
                'median_duration': trades_df['duration_hours'].median(),
            })
            
            # Profit Factor
            total_wins = winning_trades['profit'].sum() if not winning_trades.empty else 0.0
            total_losses = abs(losing_trades['profit'].sum()) if not losing_trades.empty else 0.0
            metrics['profit_factor'] = safe_divide(total_wins, total_losses, 0.0)
            
            # Calmar Ratio
            metrics['calmar_ratio'] = safe_divide(
                metrics['return_pct'], metrics['max_drawdown'], 0.0
            )
            
            # Erweiterte Risiko-Metriken
            metrics['expectancy'] = calculate_expectancy(trades_df)
            metrics['kelly_criterion'] = calculate_kelly_criterion(trades_df)
            
            # Konsistenz-Metriken
            if len(trades_df) >= 10:
                monthly_returns = calculate_monthly_returns(trades_df)
                if not monthly_returns.empty:
                    metrics['monthly_std'] = monthly_returns.std()
                    metrics['positive_months'] = (monthly_returns > 0).sum()
                    metrics['negative_months'] = (monthly_returns < 0).sum()
        
        return metrics
    except Exception as e:
        logging.error(f"Fehler bei erweiterten Metriken: {e}")
        return calculate_metrics(trades_df, equity_df, initial_capital)

def calculate_expectancy(trades_df: pd.DataFrame) -> float:
    """Erwartungswert pro Trade"""
    try:
        if trades_df.empty:
            return 0.0
        
        win_rate = safe_divide(len(trades_df[trades_df['profit'] > 0]), len(trades_df), 0.0)
        avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean() or 0.0
        avg_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].mean()) or 0.0
        
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    except Exception:
        return 0.0

def calculate_kelly_criterion(trades_df: pd.DataFrame) -> float:
    """Kelly-Kriterium für optimale Positionsgröße"""
    try:
        if trades_df.empty:
            return 0.0
        
        wins = trades_df[trades_df['profit'] > 0]['profit']
        losses = trades_df[trades_df['profit'] < 0]['profit']
        
        if wins.empty or losses.empty:
            return 0.0
        
        win_rate = len(wins) / len(trades_df)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        if avg_win <= 0:
            return 0.0
        
        return safe_divide(win_rate * avg_win - (1 - win_rate) * avg_loss, avg_win, 0.0)
    except Exception:
        return 0.0

def calculate_monthly_returns(trades_df: pd.DataFrame) -> pd.Series:
    """Monatliche Renditen berechnen"""
    try:
        if trades_df.empty or 'entry_time' not in trades_df.columns:
            return pd.Series()
        
        trades_df = trades_df.copy()
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
        
        monthly = trades_df.groupby('month')['profit'].sum()
        return monthly
    except Exception:
        return pd.Series()

def create_performance_heatmap(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Performance-Heatmap nach Zeit"""
    try:
        if trades_df.empty:
            return pd.DataFrame()
        
        trades_df = trades_df.copy()
        if 'entry_time' not in trades_df.columns:
            return pd.DataFrame()
        
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['weekday_name'] = trades_df['entry_time'].dt.day_name()
        trades_df['hour'] = trades_df['entry_time'].dt.hour
        
        heatmap = trades_df.pivot_table(
            values='profit', 
            index='weekday_name', 
            columns='hour', 
            aggfunc='sum'
        ).fillna(0)
        
        return heatmap
    except Exception as e:
        logging.warning(f"Fehler bei Heatmap-Erstellung: {e}")
        return pd.DataFrame()

# === EXPORT & DASHBOARD ===
def export_comprehensive_results(global_trades: pd.DataFrame, global_equity: pd.DataFrame, 
                               all_metrics: List[Dict], timestamp: str) -> Dict[str, str]:
    """Umfassender Export mit Error-Handling"""
    try:
        results = {}
        
        # CSV-Export
        if not global_trades.empty:
            csv_file = f"results/complete_trades_{timestamp}.csv"
            global_trades.to_csv(csv_file, index=False)
            results['csv'] = csv_file
        
        # Excel-Export
        excel_file = f"results/complete_analysis_{timestamp}.xlsx"
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                if not global_trades.empty:
                    # Hauptdaten
                    global_trades.to_excel(writer, sheet_name='Einzeltrades', index=False)
                    
                    # Equity-Verlauf
                    equity_curve_df = global_equity[['timestamp', 'symbol', 'capital']].copy()
                    equity_curve_df['timestamp'] = pd.to_datetime(equity_curve_df['timestamp'])
                    equity_curve_df.to_excel(writer, sheet_name='Equity_Verlauf', index=False)
                    
                    # Performance-Analysen
                    heatmap = create_performance_heatmap(global_trades)
                    if not heatmap.empty:
                        heatmap.to_excel(writer, sheet_name='Performance_Heatmap')
                    
                    # Zeit-Analysen
                    if {'hour', 'weekday'}.issubset(global_trades.columns):
                        try:
                            global_trades_copy = global_trades.copy()
                            global_trades_copy['entry_time'] = pd.to_datetime(global_trades_copy['entry_time'])
                            global_trades_copy['hour'] = global_trades_copy['entry_time'].dt.hour
                            global_trades_copy['weekday'] = global_trades_copy['entry_time'].dt.weekday
                            
                            time_analysis = global_trades_copy.groupby(['hour', 'weekday']).agg({
                                'profit': ['sum', 'count', 'mean'], 
                                'return_pct': 'mean'
                            })
                            time_analysis.columns = ['Profit_Sum', 'Trade_Count', 'Profit_Mean', 'Return_Mean']
                            time_analysis.to_excel(writer, sheet_name='Zeit_Analysen')
                        except Exception as e:
                            logging.warning(f"Fehler bei Zeit-Analyse: {e}")
                    
                    # Symbol-Performance
                    if 'symbol' in global_trades.columns:
                        try:
                            symbol_analysis = global_trades.groupby('symbol').agg({
                                'profit': ['sum', 'count', 'mean', 'std'],
                                'return_pct': ['mean', 'std'],
                                'duration_hours': 'mean',
                                'risk_reward_ratio': 'mean'
                            })
                            symbol_analysis.columns = ['_'.join(col) for col in symbol_analysis.columns]
                            symbol_analysis.to_excel(writer, sheet_name='Symbol_Performance')
                        except Exception as e:
                            logging.warning(f"Fehler bei Symbol-Analyse: {e}")
                    
                    # Weitere Analysen...
                    for analysis_name, column in [
                        ('Regime_Analyse', 'regime'),
                        ('Pyramid_Analyse', 'pyramid_levels'),
                        ('Exit_Analysen', 'exit_reason')
                    ]:
                        if column in global_trades.columns:
                            try:
                                analysis = global_trades.groupby(column).agg({
                                    'profit': ['sum', 'count', 'mean'], 
                                    'return_pct': 'mean'
                                })
                                analysis.columns = ['Profit_Sum', 'Trade_Count', 'Profit_Mean', 'Return_Mean']
                                analysis.to_excel(writer, sheet_name=analysis_name)
                            except Exception as e:
                                logging.warning(f"Fehler bei {analysis_name}: {e}")
                
                # Zusammenfassung
                if all_metrics:
                    summary_df = pd.DataFrame(all_metrics)
                    summary_df.to_excel(writer, sheet_name='Zusammenfassung', index=False)
            
            results['excel'] = excel_file
        except Exception as e:
            logging.error(f"Fehler beim Excel-Export: {e}")
        
        # HTML-Dashboard
        try:
            html_file = create_advanced_dashboard(global_trades, global_equity, all_metrics, timestamp)
            if html_file:
                results['html'] = html_file
        except Exception as e:
            logging.error(f"Fehler beim Dashboard-Export: {e}")
        
        return results
    except Exception as e:
        logging.error(f"Kritischer Fehler beim Export: {e}")
        return {}

def create_advanced_dashboard(trades_df: pd.DataFrame, equity_df: pd.DataFrame, 
                            all_metrics: List[Dict], timestamp: str) -> Optional[str]:
    """Erweiterte Dashboard-Erstellung mit robustem Error-Handling"""
    try:
        if trades_df.empty:
            logging.warning("Keine Trades für Dashboard verfügbar")
            return None
        
        # Daten vorbereiten
        trades_df = trades_df.copy()
        equity_df = equity_df.copy()
        
        # Timestamps normalisieren
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # Portfolio-Equity aggregieren
        portfolio_equity = equity_df.groupby('timestamp', as_index=False).agg({
            'capital': 'sum',
            'unrealized_pnl': 'sum'
        })
        portfolio_equity.rename(columns={'capital': 'portfolio_capital'}, inplace=True)
        
        # Dashboard erstellen
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Portfolio Equity (€)', 'Asset Performance', 'Monatliche Performance',
                'Trade Dauer Distribution', 'Profit/Loss Distribution', 'Regime Performance',
                'Zeit Heatmap', 'Risk-Reward Analysis', 'Exit-Gründe',
                'Drawdown-Verlauf', 'Performance Attribution', 'Monte Carlo Simulation'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "histogram"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "histogram"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Portfolio Equity Verlauf
        fig.add_trace(
            go.Scatter(
                x=portfolio_equity['timestamp'],
                y=portfolio_equity['portfolio_capital'],
                mode='lines+markers',
                name='Portfolio-Kapital',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=4),
                hovertemplate='<b>%{x}</b><br>Kapital: %{y:,.2f} €<br><extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Asset Performance
        if 'symbol' in trades_df.columns:
            asset_summary = trades_df.groupby('symbol').agg({
                'profit': 'sum',
                'symbol': 'count'
            }).rename(columns={'symbol': 'trade_count'})
            
            colors = ['#A23B72' if x < 0 else '#F18F01' for x in asset_summary['profit']]
            fig.add_trace(
                go.Bar(
                    x=asset_summary.index,
                    y=asset_summary['profit'],
                    name='Asset Profit',
                    marker_color=colors,
                    text=[f"{count} Trades" for count in asset_summary['trade_count']],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Profit: %{y:,.2f} €<br>Trades: %{text}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Monatliche Performance
        try:
            monthly = trades_df.groupby(trades_df['entry_time'].dt.to_period('M'))['profit'].sum()
            colors_monthly = ['#C73E1D' if x < 0 else '#3F7D20' for x in monthly.values]
            
            fig.add_trace(
                go.Bar(
                    x=[str(x) for x in monthly.index],
                    y=monthly.values,
                    name='Monatlich',
                    marker_color=colors_monthly,
                    hovertemplate='<b>%{x}</b><br>P&L: %{y:,.2f} €<extra></extra>'
                ),
                row=1, col=3
            )
        except Exception as e:
            logging.warning(f"Fehler bei monatlicher Performance: {e}")
        
        # 4. Trade Duration Distribution
        if 'duration_hours' in trades_df.columns:
            fig.add_trace(
                go.Histogram(
                    x=trades_df['duration_hours'],
                    name='Dauer',
                    nbinsx=30,
                    marker_color='#6A994E',
                    opacity=0.7,
                    hovertemplate='Dauer: %{x:.1f}h<br>Anzahl: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 5. Profit/Loss Distribution
        colors_profit = ['#C73E1D' if x < 0 else '#3F7D20' for x in trades_df['profit']]
        fig.add_trace(
            go.Histogram(
                x=trades_df['profit'],
                name='P&L Distribution',
                nbinsx=40,
                marker=dict(color=colors_profit, opacity=0.7),
                hovertemplate='Profit: %{x:,.2f} €<br>Anzahl: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 6. Regime Performance
        if 'regime' in trades_df.columns:
            regime_perf = trades_df.groupby('regime')['profit'].sum()
            fig.add_trace(
                go.Bar(
                    x=regime_perf.index,
                    y=regime_perf.values,
                    name='Regime',
                    marker_color='#BC6C25',
                    hovertemplate='<b>%{x}</b><br>P&L: %{y:,.2f} €<extra></extra>'
                ),
                row=2, col=3
            )
        
        # 7. Zeit Heatmap
        try:
            heatmap_data = create_performance_heatmap(trades_df)
            if not heatmap_data.empty:
                # Wochentage in richtige Reihenfolge bringen
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_data = heatmap_data.reindex([d for d in day_order if d in heatmap_data.index])
                
                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_data.values,
                        x=heatmap_data.columns,
                        y=heatmap_data.index,
                        colorscale='RdYlGn',
                        name='Zeit Heatmap',
                        hovertemplate='Stunde: %{x}<br>Tag: %{y}<br>P&L: %{z:,.2f} €<extra></extra>'
                    ),
                    row=3, col=1
                )
        except Exception as e:
            logging.warning(f"Fehler bei Zeit-Heatmap: {e}")
        
        # 8. Risk-Reward Analysis
        if 'risk_reward_ratio' in trades_df.columns:
            profit_colors = ['#C73E1D' if x < 0 else '#3F7D20' for x in trades_df['profit']]
            
            fig.add_trace(
                go.Scatter(
                    x=trades_df['risk_reward_ratio'],
                    y=trades_df['profit'],
                    mode='markers',
                    name='Risk-Reward',
                    marker=dict(
                        color=profit_colors,
                        size=8,
                        opacity=0.6,
                        line=dict(width=1, color='white')
                    ),
                    customdata=trades_df[['symbol', 'entry_time', 'exit_time', 'size', 'exit_reason', 'regime']].values,
                    hovertemplate=(
                        '<b>%{customdata[0]}</b><br>'
                        'Entry: %{customdata[1]}<br>'
                        'Exit: %{customdata[2]}<br>'
                        'Size: %{customdata[3]:,.0f} €<br>'
                        'Profit: %{y:,.2f} €<br>'
                        'R/R: %{x:.2f}<br>'
                        'Exit: %{customdata[4]}<br>'
                        'Regime: %{customdata[5]}<extra></extra>'
                    )
                ),
                row=3, col=2
            )
        
        # 9. Exit-Gründe
        if 'exit_reason' in trades_df.columns:
            exit_reasons = trades_df.groupby('exit_reason').agg({
                'profit': 'sum',
                'exit_reason': 'count'
            }).rename(columns={'exit_reason': 'count'})
            
            fig.add_trace(
                go.Bar(
                    x=exit_reasons.index,
                    y=exit_reasons['profit'],
                    name='Exit-Gründe',
                    marker_color='#6F1D1B',
                    text=[f"{count} Trades" for count in exit_reasons['count']],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>P&L: %{y:,.2f} €<br>Anzahl: %{text}<extra></extra>'
                ),
                row=3, col=3
            )
        
        # 10. Drawdown-Verlauf
        try:
            running_max = portfolio_equity['portfolio_capital'].expanding().max()
            drawdown = (running_max - portfolio_equity['portfolio_capital']) / running_max * 100
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio_equity['timestamp'],
                    y=drawdown,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='#C73E1D', width=2),
                    fill='tonexty',
                    fillcolor='rgba(199, 62, 29, 0.3)',
                    hovertemplate='%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
                ),
                row=4, col=1
            )
        except Exception as e:
            logging.warning(f"Fehler bei Drawdown-Berechnung: {e}")
        
        # 11. Performance Attribution
        if 'pyramid_levels' in trades_df.columns:
            pyramid_perf = trades_df.groupby('pyramid_levels').agg({
                'profit': 'sum',
                'pyramid_levels': 'count'
            }).rename(columns={'pyramid_levels': 'count'})
            
            fig.add_trace(
                go.Bar(
                    x=[f"Level {x}" for x in pyramid_perf.index],
                    y=pyramid_perf['profit'],
                    name='Pyramid Levels',
                    marker_color='#99582A',
                    text=[f"{count} Trades" for count in pyramid_perf['count']],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>P&L: %{y:,.2f} €<br>Trades: %{text}<extra></extra>'
                ),
                row=4, col=2
            )
        
        # 12. Monte Carlo Simulation
        try:
            if len(trades_df) >= 10 and 'return_pct' in trades_df.columns:
                returns = trades_df['return_pct'].dropna()
                if len(returns) > 0:
                    avg_return = returns.mean() / 100  # Convert to decimal
                    std_return = returns.std() / 100
                    
                    # Monte Carlo mit 1000 Simulationen
                    np.random.seed(42)  # Für reproduzierbare Ergebnisse
                    mc_results = []
                    
                    for _ in range(1000):
                        random_returns = np.random.normal(avg_return, std_return, len(returns))
                        final_capital = 10000 * np.prod(1 + random_returns)
                        mc_results.append(final_capital)
                    
                    fig.add_trace(
                        go.Histogram(
                            x=mc_results,
                            name='Monte Carlo',
                            nbinsx=30,
                            marker_color='#386641',
                            opacity=0.7,
                            hovertemplate='Endkapital: %{x:,.0f} €<br>Häufigkeit: %{y}<extra></extra>'
                        ),
                        row=4, col=3
                    )
        except Exception as e:
            logging.warning(f"Fehler bei Monte Carlo Simulation: {e}")
        
        # Layout anpassen
        fig.update_layout(
            title={
                'text': f"🚀 Advanced Trading Dashboard | {timestamp}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2E86AB'}
            },
            height=1800,
            showlegend=False,
            template='plotly_white',
            font=dict(size=10),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Achsen-Formatierung
        for i in range(1, 5):
            for j in range(1, 4):
                fig.update_xaxes(tickangle=45, row=i, col=j)
                fig.update_yaxes(title_font_size=10, row=i, col=j)
        
        # HTML speichern
        html_file = f"results/enhanced_dashboard_{timestamp}.html"
        fig.write_html(
            html_file,
            config={'displayModeBar': True, 'responsive': True}
        )
        
        logging.info(f"Dashboard erfolgreich erstellt: {html_file}")
        return html_file
        
    except Exception as e:
        logging.error(f"Kritischer Fehler bei Dashboard-Erstellung: {e}")
        logging.error(traceback.format_exc())
        return None

# === HAUPTFUNKTION ===
def main_complete():
    """Hauptfunktion mit vollständigem Error-Handling"""
    try:
        setup_logging()
        logging.info("=== STARTE ERWEITERTE TRADING-STRATEGIE ===")
        
        # Pfad-Management
        last_path = os.path.join(os.getcwd(), 'data', 'raw')
        if os.path.exists(os.path.join('results', 'last_path.json')):
            try:
                with open(os.path.join('results', 'last_path.json'), 'r') as f:
                    last_path = json.load(f).get('last_path', last_path)
            except Exception:
                pass
        
        # Dateiauswahl
        file_paths = select_files(last_path)
        if not file_paths:
            print("❌ Keine Dateien ausgewählt. Beende Programm.")
            return
        
        # Initialisierung
        all_trades = []
        all_equity = []
        all_metrics = []
        total_initial_capital = len(file_paths) * 10000  # 10k pro Asset
        
        print(f"🚀 Starte Analyse von {len(file_paths)} Assets mit je 10.000€ Startkapital...")
        print(f"📊 Gesamtes Startkapital: {total_initial_capital:,.0f}€")
        
        # Asset-Verarbeitung
        successful_assets = 0
        failed_assets = 0
        
        for file_path in tqdm(file_paths, desc="Verarbeite Assets"):
            try:
                # Datei laden
                df = pd.read_parquet(file_path, engine='pyarrow')
                
                # Validierung
                required_columns = ['open', 'high', 'low', 'close']
                if not all(col in df.columns for col in required_columns):
                    logging.warning(f"Überspringe {os.path.basename(file_path)} - fehlende OHLC-Spalten")
                    failed_assets += 1
                    continue
                
                if len(df) < 100:
                    logging.warning(f"Überspringe {os.path.basename(file_path)} - zu wenig Daten ({len(df)} Zeilen)")
                    failed_assets += 1
                    continue
                
                # Symbol extrahieren
                symbol = os.path.basename(file_path).replace('.parquet', '').split('_')[0].upper()
                df['symbol'] = symbol
                
                # Timestamp-Management
                if not isinstance(df.index, pd.DatetimeIndex):
                    start_date = pd.to_datetime('2024-01-01 00:00:00')
                    df['timestamp'] = [start_date + timedelta(hours=i) for i in range(len(df))]
                    df.set_index('timestamp', inplace=True)
                else:
                    df['timestamp'] = df.index
                
                # Indikatoren berechnen
                df = calculate_advanced_indicators(df)
                
                # Pattern-Erkennung
                patterns = detect_candlestick_patterns(df)
                
                if patterns.empty:
                    logging.warning(f"Keine Patterns erkannt für {symbol}")
                    failed_assets += 1
                    continue
                
                # Strategie simulieren
                trades_df, equity_df, final_capital = simulate_advanced_strategy(
                    df, patterns, initial_capital=10000
                )
                
                if not trades_df.empty:
                    all_trades.append(trades_df)
                    all_equity.append(equity_df)
                    
                    # Metriken berechnen
                    metrics = calculate_enhanced_metrics(trades_df, equity_df, 10000)
                    metrics['symbol'] = symbol
                    all_metrics.append(metrics)
                    
                    # Fortschritt anzeigen
                    profit_loss = final_capital - 10000
                    return_pct = (profit_loss / 10000) * 100
                    print(f"✅ {symbol:>8}: {len(trades_df):>3} Trades | "
                          f"{return_pct:>6.1f}% | "
                          f"{final_capital:>10,.0f}€")
                    
                    successful_assets += 1
                else:
                    logging.warning(f"Keine Trades generiert für {symbol}")
                    failed_assets += 1
            
            except Exception as e:
                logging.error(f"Fehler bei {os.path.basename(file_path)}: {e}")
                failed_assets += 1
                continue
        
        # Ergebnisse zusammenfassen
        if not all_trades:
            print("❌ Keine erfolgreichen Trades generiert!")
            return
        
        # DataFrames kombinieren
        global_trades = pd.concat(all_trades, ignore_index=True)
        global_equity = pd.concat(all_equity, ignore_index=True)
        
        # Portfolio-Zusammenfassung
        total_trades = len(global_trades)
        total_profit = global_trades['profit'].sum()
        final_portfolio_value = total_initial_capital + total_profit
        portfolio_return = (total_profit / total_initial_capital) * 100
        
        print("\n" + "="*60)
        print("📈 PORTFOLIO-ZUSAMMENFASSUNG")
        print("="*60)
        print(f"✅ Erfolgreiche Assets:    {successful_assets:>3}")
        print(f"❌ Fehlgeschlagene Assets: {failed_assets:>3}")
        print(f"📊 Gesamt Trades:          {total_trades:>3}")
        print(f"💰 Startkapital:           {total_initial_capital:>10,.0f}€")
        print(f"💰 Endkapital:             {final_portfolio_value:>10,.0f}€")
        print(f"📈 Gewinn/Verlust:         {total_profit:>10,.0f}€")
        print(f"📊 Gesamtrendite:          {portfolio_return:>9.2f}%")
        
        if all_metrics:
            avg_win_rate = np.mean([m.get('win_rate', 0) for m in all_metrics])
            avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in all_metrics])
            max_drawdown = max([m.get('max_drawdown', 0) for m in all_metrics])
            
            print(f"🎯 Durchschnittliche Win-Rate: {avg_win_rate*100:>6.1f}%")
            print(f"📊 Durchschnittliche Sharpe:   {avg_sharpe:>9.2f}")
            print(f"📉 Maximaler Drawdown:         {max_drawdown:>9.2f}%")
        
        print("="*60)
        
        # Export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = export_comprehensive_results(global_trades, global_equity, all_metrics, timestamp)
        
        print("\n📁 EXPORTIERTE DATEIEN:")
        for file_type, file_path in results.items():
            if file_path and os.path.exists(file_path):
                print(f"  {file_type.upper():>5}: {file_path}")
        
        print(f"\n🎉 Analyse abgeschlossen! Ergebnisse in './results/' Ordner.")
        logging.info("=== TRADING-STRATEGIE ERFOLGREICH ABGESCHLOSSEN ===")
        
    except Exception as e:
        logging.error(f"Kritischer Fehler in main_complete(): {e}")
        logging.error(traceback.format_exc())
        print(f"❌ Kritischer Fehler: {e}")
    
    finally:
        # Cleanup
        logging.info("Cleanup abgeschlossen")

# === PROGRAMM-EINSTIEG ===
if __name__ == "__main__":
    try:
        main_complete()
    except KeyboardInterrupt:
        print("\n\n⚠️  Programm durch Benutzer abgebrochen!")
        logging.info("Programm durch Benutzer abgebrochen")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        logging.error(f"Unerwarteter Fehler: {e}")
        logging.error(traceback.format_exc())
    finally:
        print("\nProgramm beendet.")
        input("Drücke Enter zum Schließen...")  # Konsole offen halten