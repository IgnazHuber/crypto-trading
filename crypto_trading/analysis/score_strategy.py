# crypto_trading/analysis/score_strategy.py

import pandas as pd

def compute_score(row, weights, triggers):
    score = 0
    reasons = []
    for ind, weight in weights.items():
        val = row.get(ind, None)
        if val is None or pd.isna(val):
            continue
        t = triggers.get(ind, None)
        if t:
            if isinstance(t, (tuple, list)) and t[0] <= val <= t[1]:
                score += weight
                reasons.append(f"{ind} ({val:.2f}) im Trigger [{t[0]}, {t[1]}]")
            elif isinstance(t, dict):
                if t.get("long") and t["long"][0] <= val <= t["long"][1]:
                    score += weight
                    reasons.append(f"{ind} ({val:.2f}) Long-Trigger")
                elif t.get("short") and t["short"][0] <= val <= t["short"][1]:
                    score -= weight
                    reasons.append(f"{ind} ({val:.2f}) Short-Trigger")
        else:
            score += weight
    return score, reasons

def score_strategy(df, weights, triggers, long_threshold=1.5, short_threshold=-1.5):
    signals = []
    all_scores = []
    details = []
    for _, row in df.iterrows():
        score, reason = compute_score(row, weights, triggers)
        all_scores.append(score)
        details.append(reason)
        if score >= long_threshold:
            signals.append(1)
        elif score <= short_threshold:
            signals.append(-1)
        else:
            signals.append(0)
    df["score"] = all_scores
    df["score_details"] = details
    df["score_signal"] = signals
    return df

def run_score_strategy(df, weights, triggers, long_threshold=1.5, short_threshold=-1.5):
    return score_strategy(df, weights, triggers, long_threshold, short_threshold)
