# pattern_detection_250805a.py
SKRIPT_ID = "pattern_detection_250805a"
"""
Candlestick-Pattern und Signalgeber für das Krypto-Framework.
Enthält alle neuen Strategiefunktionen inkl. Benchmark ("always_trade").
"""

import pandas as pd

# Klassische Candlestick-Pattern
def bullish_engulfing(df):
    return (
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['close'] > df['open']) &
        (df['close'] > df['open'].shift(1)) &
        (df['open'] < df['close'].shift(1))
    )

def bearish_engulfing(df):
    return (
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['close'] < df['open']) &
        (df['open'] > df['close'].shift(1)) &
        (df['close'] < df['open'].shift(1))
    )

def morning_star(df):
    return (
        (df['close'].shift(2) < df['open'].shift(2)) &
        ((df['close'].shift(1) - df['open'].shift(1)).abs() < 0.3 * (df['high'].shift(1) - df['low'].shift(1))) &
        (df['close'] > df['open']) &
        (df['close'] > ((df['open'].shift(2) + df['close'].shift(2))/2))
    )

def shooting_star(df):
    body = (df['close'] - df['open']).abs()
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
    return (upper_shadow > 2 * body) & (lower_shadow < body) & (body > 0.2 * (df['high'] - df['low']))

def hammer(df):
    body = (df['close'] - df['open']).abs()
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    return (lower_shadow > 2 * body) & (upper_shadow < body) & (body > 0.2 * (df['high'] - df['low']))

def evening_star(df):
    return (
        (df['close'].shift(2) > df['open'].shift(2)) &
        ((df['close'].shift(1) - df['open'].shift(1)).abs() < 0.3 * (df['high'].shift(1) - df['low'].shift(1))) &
        (df['close'] < df['open']) &
        (df['close'] < ((df['open'].shift(2) + df['close'].shift(2))/2))
    )

def inside_bar(df):
    return (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))

def doji(df):
    return (abs(df['close'] - df['open']) < 0.1 * (df['high'] - df['low']))

def piercing_line(df):
    return (
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['open'] < df['low'].shift(1)) &
        (df['close'] > (df['open'].shift(1) + df['close'].shift(1))/2) &
        (df['close'] < df['open'].shift(1))
    )

def three_white_soldiers(df):
    return (
        (df['close'] > df['open']) &
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['close'].shift(2) > df['open'].shift(2))
    )

# Neue Signal-Pattern für Strategien
def bollinger_squeeze_long(df):
    bb_width = df['bb_width']
    squeeze = bb_width < bb_width.rolling(100).quantile(0.10)
    breakout_long = df['close'] > df['bb_upper']
    return squeeze & breakout_long

def bollinger_squeeze_short(df):
    bb_width = df['bb_width']
    squeeze = bb_width < bb_width.rolling(100).quantile(0.10)
    breakout_short = df['close'] < df['bb_lower']
    return squeeze & breakout_short

def vwap_breakout_long(df):
    return df['close'] > df['vwap']

def vwap_breakout_short(df):
    return df['close'] < df['vwap']

def mfi_long(df):
    return df['mfi'] < 20

def mfi_short(df):
    return df['mfi'] > 80

def heikin_ashi_trend_long(df):
    return df['ha_close'] > df['ha_open']

def heikin_ashi_trend_short(df):
    return df['ha_close'] < df['ha_open']

def atr_breakout_long(df):
    return df['close'] > (df['high'] + df['atr'])

def atr_breakout_short(df):
    return df['close'] < (df['low'] - df['atr'])

def ema4h_crossover_long(df):
    return df['close'] > df['ema4h']

def ema4h_crossover_short(df):
    return df['close'] < df['ema4h']

# BENCHMARK: Viele Trades, jede gerade Stunde
def always_trade(df):
    if 'timestamp' in df.columns:
        return (pd.to_datetime(df['timestamp']).dt.hour % 2 == 0)
    else:
        return pd.Series([False]*len(df), index=df.index)

PATTERN_FUNCTIONS = {
    "bullish_engulfing": bullish_engulfing,
    "bearish_engulfing": bearish_engulfing,
    "morning_star": morning_star,
    "shooting_star": shooting_star,
    "hammer": hammer,
    "evening_star": evening_star,
    "inside_bar": inside_bar,
    "doji": doji,
    "piercing_line": piercing_line,
    "three_white_soldiers": three_white_soldiers,
    "bollinger_squeeze_long": bollinger_squeeze_long,
    "bollinger_squeeze_short": bollinger_squeeze_short,
    "vwap_breakout_long": vwap_breakout_long,
    "vwap_breakout_short": vwap_breakout_short,
    "mfi_long": mfi_long,
    "mfi_short": mfi_short,
    "heikin_ashi_trend_long": heikin_ashi_trend_long,
    "heikin_ashi_trend_short": heikin_ashi_trend_short,
    "atr_breakout_long": atr_breakout_long,
    "atr_breakout_short": atr_breakout_short,
    "ema4h_crossover_long": ema4h_crossover_long,
    "ema4h_crossover_short": ema4h_crossover_short,
    "always_trade": always_trade,  # Benchmark/Many-Trades-Strategie
}

EXPECTED_KEYS = list(PATTERN_FUNCTIONS.keys())
for k in EXPECTED_KEYS:
    if k not in PATTERN_FUNCTIONS:
        print(f"[{SKRIPT_ID}] WARNUNG: Pattern '{k}' fehlt im PATTERN_FUNCTIONS!")

if __name__ == "__main__":
    print(f"[{SKRIPT_ID}] Test: Pattern Detection ready.")
