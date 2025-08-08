# multi_strategy_score_250805a.py
SKRIPT_ID = "multi_strategy_score_250805a"
"""
Meta-Signal/Score-Generator für Multi-Strategie-Backtests.
"""

import pandas as pd

def generate_score_signals(df, regimes, strategy_config, pattern_functions, min_signals=2):
    """
    Generiert für jede Zeile einen Score (Anzahl aktiver Strategien).
    Args:
        df: Kurs- und Indikator-DataFrame.
        regimes: Regime-DataFrame.
        strategy_config: Liste der Strategien wie im Framework.
        pattern_functions: Dict aller Pattern-Funktionen.
        min_signals: Mindestanzahl gleichzeitiger Signale für Entry.
    Returns:
        score_df: DataFrame mit Einzelstrategie- und Meta-Signalen.
    """
    signals = []
    for strat_name, candle_key, regime_key, regime_desc in strategy_config:
        candle_func = pattern_functions[candle_key]
        regime_mask = regimes[regime_key] if regime_key in regimes else pd.Series([False]*len(df), index=df.index)
        signal = (candle_func(df) & regime_mask).astype(int)
        signals.append(signal.rename(strat_name))
    score_df = pd.concat(signals, axis=1)
    score_df['score_long'] = score_df.sum(axis=1)
    score_df['meta_long'] = (score_df['score_long'] >= min_signals).astype(int)
    return score_df
