# analysis/mod_strategy.py

from .pattern_detection_250805a import PATTERN_FUNCTIONS
from .market_regime_250805a import classify_market_regime
from .multi_strategy_score_250805a import generate_score_signals
from .mod_indicators import calc_indicators

STRATEGY_CONFIG = [
    ("Many Trades Strategy", "always_trade", "trend", "Jede gerade Stunde (Demo)"),
    ("Bollinger Squeeze + Ausbruch Long", "bollinger_squeeze_long", "squeeze_breakout_long", "Squeeze + Breakout Long"),
    ("VWAP Breakout Long", "vwap_breakout_long", "uptrend", "Kurs > VWAP im Aufwärtstrend"),
    # ... weitere Strategien ...
]

def prepare_regimes(df):
    df = calc_indicators(df)
    print("Spalten vor Regime-Berechnung:", df.columns.tolist())
    regimes_result = classify_market_regime(df)
    import pandas as pd
    if isinstance(regimes_result, pd.DataFrame):
        return regimes_result
    elif isinstance(regimes_result, pd.Series):
        return regimes_result.to_frame()
    elif isinstance(regimes_result, dict):
        return pd.DataFrame(regimes_result, index=df.index)
    else:
        return pd.DataFrame(index=df.index)

def generate_signals(df, regimes, params):
    return generate_score_signals(df, regimes, STRATEGY_CONFIG, PATTERN_FUNCTIONS, min_signals=params["min_signals"])
