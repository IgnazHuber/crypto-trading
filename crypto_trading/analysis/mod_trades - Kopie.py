import os
import json
import pandas as pd
from .indicator_context import classify_trend, classify_volatility, classify_volume, apply_context_to_indicators

def generate_signals(df, params):
    """
    Erzeugt Handelssignale basierend auf Indikatoren.
    Beispiel-Logik (Platzhalter, erweiterbar):
      - Long-Signal, wenn RSI < oversold und EMA20 > SMA50
      - Dummy-PnL = +2% des Entry-Preises
    :param df: DataFrame mit Indikator-Spalten
    :param params: Dictionary mit Parametern, erwartet ggf. "oversold"
    :return: Liste von Signalen (Liste von Dicts mit 'pnl_abs')
    """
    signals = []
    oversold = params.get("oversold", 30)
    ema_cols = [c for c in df.columns if "ema_20" in c]
    sma_cols = [c for c in df.columns if "sma_50" in c]
    rsi_cols = [c for c in df.columns if "rsi_14" in c]

    if ema_cols and sma_cols and rsi_cols:
        ema_col = ema_cols[0]
        sma_col = sma_cols[0]
        rsi_col = rsi_cols[0]
        for _, row in df.iterrows():
            if row[rsi_col] < oversold and row[ema_col] > row[sma_col]:
                # Dummy-PnL: +2% vom aktuellen Kurs
                signals.append({"pnl_abs": row["close"] * 0.02})
    return signals


def run_meta_strategy_with_indicators(df, strategy_name, params, trade_every=1,
                                      asset="ASSET", start_capital=10000.0,
                                      max_risk_per_trade=0.10, sl_pct=0.03, tp_pct=0.20):
    """
    Meta-Strategie:
    - erkennt Marktumfeld (Trend, Volatilität, Volumen)
    - lädt Parameter je Marktumfeld aus best_params.json
    - erzeugt Signale mit passenden Parametern
    - führt Trades mit Kapitalmanagement aus
    Rückgabe:
    - DataFrame mit allen Trades (inkl. Kapital)
    - Endkapital
    - Maximaler Einzelverlust
    """
    # --- Parameter laden ---
    results_file = os.path.join("results", "best_params.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            best_params_map = json.load(f)
    else:
        best_params_map = {"sideways": params, "uptrend": params, "downtrend": params}

    # --- Marktumfeld berechnen (falls nicht vorhanden) ---
    if "trend_class" not in df.columns:
        df = classify_trend(df)
        df = classify_volatility(df)
        df = classify_volume(df)
        df = apply_context_to_indicators(df)

    capital = start_capital
    max_loss = 0.0
    trades = []

    # --- Tradesimulation ---
    for idx in range(0, len(df), trade_every):
        row = df.iloc[idx]
        market_cond = row["trend_class"]
        cond_params = best_params_map.get(market_cond, params)

        # Signale mit marktbedingten Parametern
        signals = generate_signals(df.iloc[:idx+1], cond_params)
        if not signals:
            continue

        # Letztes Signal = Tradeentscheidung
        sig = signals[-1]
        trade_size = capital * max_risk_per_trade
        raw_pnl = sig.get("pnl_abs", 0.0)
        sl_limit = -trade_size * sl_pct
        tp_limit = trade_size * tp_pct
        pnl = max(min(raw_pnl, tp_limit), sl_limit)

        capital += pnl
        max_loss = min(max_loss, pnl)
        trades.append({
            "time": row.name,
            "market_cond": market_cond,
            "trade_size": trade_size,
            "pnl": pnl,
            "capital": capital
        })

    return pd.DataFrame(trades), capital, max_loss
