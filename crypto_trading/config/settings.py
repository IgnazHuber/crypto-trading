# config/settings.py

import os
import re

# === Globale Einstellungen ===
START_CAPITAL = 10_000
ASSET = "BTCUSDT"
MIN_CAPITAL = 100

BASE_DIR = r"d:\Projekte\crypto_trading"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PARQUET_PATH = r"D:\Projekte\crypto_trading\crypto_trading\data\raw\BTCUSDT_1h_1year_ccxt.parquet"
#PARQUET_PATH = os.path.join(DATA_DIR, "BTCUSDT_1h_1year_ccxt.parquet")
OUTPUT_EXCEL = os.path.join(RESULTS_DIR, "btc_candlestick_backtest_regime.xlsx")

# Strategie-Konfiguration
STRATEGY_CONFIG = [
    ("Bullish Engulfing + Aufwärtstrend", "bullish_engulfing", "uptrend", "EMA50>EMA200 & MACD>0"),
    ("Bearish Engulfing + Abwärtstrend", "bearish_engulfing", "downtrend", "EMA50<EMA200 & MACD<0"),
    ("Morning Star + High Volatility", "morning_star", "high_volatility", "ATR>Median"),
    ("Shooting Star + Überkauft", "shooting_star", "overbought", "RSI>70"),
    ("Hammer + Seitwärtsmarkt", "hammer", "sideways", "ADX<20"),
    ("Evening Star + BB-Upper", "evening_star", "bb_upper", "Kurs>Bollinger-Upper"),
    ("Inside Bar + Trend", "inside_bar", "trend", "Trendfilter: EMA20>EMA50>EMA200"),
    ("Doji + Überkauft/Überverkauft", "doji", "overbought_or_oversold", "RSI>80/RSI<20"),
    ("Piercing Line + Downtrend", "piercing_line", "downtrend", "EMA50<EMA200 & MACD<0"),
    ("Three White Soldiers + Breakout", "three_white_soldiers", "breakout", "BB-Width > 75%")
]

def safe_sheet_name(name, maxlen=28):
    """Excel-fähiger Sheetname, für Plots/Dateien und Excel-Sheets"""
    name = re.sub(r'[\/\\\?\*\[\]\:]', '_', name)
    return name[:maxlen]
