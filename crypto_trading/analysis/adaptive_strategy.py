# crypto_trading/analysis/adaptive_strategy.py

import pandas as pd
from crypto_trading.analysis.score_strategy import compute_score

def adaptive_score_strategy(df, regimes_weights, regimes_triggers, long_thresholds, short_thresholds):
    signals = []
    scores = []
    details = []
    for _, row in df.iterrows():
        regime = row.get("Regime", "trend")
        weights = regimes_weights.get(regime, {})
        triggers = regimes_triggers.get(regime, {})
        long_thres = long_thresholds.get(regime, 1.5)
        short_thres = short_thresholds.get(regime, -1.5)
        score, reason = compute_score(row, weights, triggers)
        scores.append(score)
        details.append(reason)
        if score >= long_thres:
            signals.append(1)
        elif score <= short_thres:
            signals.append(-1)
        else:
            signals.append(0)
    df["score"] = scores
    df["score_details"] = details
    df["score_signal"] = signals
    return df

def run_adaptive_strategy(df, regimes_weights, regimes_triggers, long_thresholds, short_thresholds):
    return adaptive_score_strategy(df, regimes_weights, regimes_triggers, long_thresholds, short_thresholds)
