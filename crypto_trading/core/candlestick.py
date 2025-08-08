# core/candlestick.py

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

CANDLE_FUNC = {
    "bullish_engulfing": bullish_engulfing,
    "bearish_engulfing": bearish_engulfing,
    "morning_star": morning_star,
    "shooting_star": shooting_star,
    "hammer": hammer,
    "evening_star": evening_star,
    "inside_bar": inside_bar,
    "doji": doji,
    "piercing_line": piercing_line,
    "three_white_soldiers": three_white_soldiers
}
