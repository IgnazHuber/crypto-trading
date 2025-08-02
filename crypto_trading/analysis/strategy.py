"""
Modul: strategy.py

Trading-Logik für Score-basierte Strategien mit beliebigen Indikatoren,
Multi-Asset/Paramset-Support, Long/Short, SL/TP, gewichtete Score-Bildung,
erweiterbar für Batch/Grid und Research.

Funktionen:
- run_score_strategy(df, params, asset, config): DataFrame aller Trades

Author: ChatGPT Research, 2025
"""

import pandas as pd
import numpy as np
import ta

DEFAULT_CONFIG = {
    "score_entry_threshold": 3,     # Mind. Score für Einstieg
    "allow_short": False,           # Short-Trading aktivieren (optional)
    "indicators": [
        # Name, Function, Args, Score Weight
        ("rsi",      lambda df, w: ta.momentum.RSIIndicator(df['close'], window=w).rsi(),          {"window": 14},   1),
        ("macd",     lambda df, f,s,g: ta.trend.MACD(df['close'], window_fast=f, window_slow=s, window_sign=g).macd(), {"window_fast": 12, "window_slow": 26, "window_sign": 9}, 1),
        ("macd_sig", lambda df, f,s,g: ta.trend.MACD(df['close'], window_fast=f, window_slow=s, window_sign=g).macd_signal(), {"window_fast": 12, "window_slow": 26, "window_sign": 9}, 0),
        ("ema_short",lambda df, w: ta.trend.EMAIndicator(df['close'], window=w).ema_indicator(),   {"window": 50},   1),
        ("ema_long", lambda df, w: ta.trend.EMAIndicator(df['close'], window=w).ema_indicator(),   {"window": 200},  1),
        ("bb_upper", lambda df, w: ta.volatility.BollingerBands(df['close'], window=w).bollinger_hband(), {"window": 20}, 0),
        ("bb_lower", lambda df, w: ta.volatility.BollingerBands(df['close'], window=w).bollinger_lband(), {"window": 20}, 1),
        ("stoch_k",  lambda df, w: ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=w).stoch(), {"window": 14}, 1),
        ("stoch_d",  lambda df, w: ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=w).stoch_signal(), {"window": 14}, 0)
    ]
}

def run_score_strategy(
        df, params, asset,
        config=DEFAULT_CONFIG,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        trade_mode='long'
    ):
    """
    Score-basierte Handelsstrategie, Multi-Asset, Batch- und Grid-fähig.
    Liefert detaillierte Einzeltrade-Tabelle (inkl. Score, Analyse, Trade-IDs).

    Args:
        df: DataFrame mit Preisdaten ('close','high','low',...)
        params: tuple/list aller Parametereinstellungen, Reihenfolge wie in GRID
        asset: Name/Shortcode des Assets (z.B. 'BTC')
        config: Dict (Indikator-Konfig & Score)
        stop_loss_pct: SL in Prozent (z.B. 0.03 = 3%)
        take_profit_pct: TP in Prozent
        trade_mode: 'long', 'short' oder 'both'
    Returns:
        DataFrame: Alle Trades
    """
    df = df.copy()
    # Entpacke Parameter, hier beispielhaft: (Reihenfolge Grid und Mapping)
    (
        RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, EMA_SHORT, EMA_LONG,
        BB_WINDOW, STOCH_WINDOW, STOP_LOSS_PCT, TAKE_PROFIT_PCT
    ) = params

    # ========== Indikatoren berechnen ==========
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=RSI_PERIOD).rsi()
    macd = ta.trend.MACD(df['close'], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema_short'] = ta.trend.EMAIndicator(df['close'], window=EMA_SHORT).ema_indicator()
    df['ema_long'] = ta.trend.EMAIndicator(df['close'], window=EMA_LONG).ema_indicator()
    bb = ta.volatility.BollingerBands(df['close'], window=BB_WINDOW)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=STOCH_WINDOW)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df = df.dropna()

    entries = []
    position = None
    entry_price = 0
    entry_score = 0
    entry_reasons = ""
    trade_dir = None
    trade_modes = []
    if trade_mode == 'both':
        trade_modes = ['long', 'short']
    else:
        trade_modes = [trade_mode]

    for i, row in df.iterrows():
        # === Score-Logik für Entry ===
        score = 0
        reasons = []
        # --- LONG ---
        long_conds = [
            (row['rsi'] < 35, "RSI < 35"),
            (row['macd'] > row['macd_signal'], "MACD Bull-Crossover"),
            (row['ema_short'] > row['ema_long'], f"EMA({EMA_SHORT}) > EMA({EMA_LONG})"),
            (row['close'] <= row['bb_lower'], "Kurs <= BB lower"),
            (row['stoch_k'] < 25 and row['stoch_k'] > row['stoch_d'], "Stoch %K < 25 & > %D")
        ]
        # --- SHORT ---
        short_conds = [
            (row['rsi'] > 65, "RSI > 65"),
            (row['macd'] < row['macd_signal'], "MACD Bear-Crossover"),
            (row['ema_short'] < row['ema_long'], f"EMA({EMA_SHORT}) < EMA({EMA_LONG})"),
            (row['close'] >= row['bb_upper'], "Kurs >= BB upper"),
            (row['stoch_k'] > 75 and row['stoch_k'] < row['stoch_d'], "Stoch %K > 75 & < %D")
        ]

        # Entry- und Exit-Bedingungen für beide Richtungen prüfen
        entries_this_bar = []
        for mode in trade_modes:
            curr_score = 0
            curr_reasons = []
            if mode == 'long':
                for cond, txt in long_conds:
                    if cond: curr_score += 1; curr_reasons.append(txt)
            elif mode == 'short':
                for cond, txt in short_conds:
                    if cond: curr_score += 1; curr_reasons.append(txt)
            # Entry ab Score
            if curr_score >= config.get("score_entry_threshold", 3):
                entries_this_bar.append((mode, curr_score, "\n".join(curr_reasons)))
        # Exit-Logik
        exit_long = (
            (row['rsi'] > 60) or
            (row['macd'] < row['macd_signal']) or
            (row['close'] >= row['bb_upper']) or
            (row['stoch_k'] > 80) or
            (row['close'] < entry_price * (1 - STOP_LOSS_PCT)) or
            (row['close'] > entry_price * (1 + TAKE_PROFIT_PCT))
        )
        exit_short = (
            (row['rsi'] < 40) or
            (row['macd'] > row['macd_signal']) or
            (row['close'] <= row['bb_lower']) or
            (row['stoch_k'] < 20) or
            (row['close'] > entry_price * (1 + STOP_LOSS_PCT)) or
            (row['close'] < entry_price * (1 - TAKE_PROFIT_PCT))
        )
        # Trades managen
        if position == 'long' and exit_long:
            pnl_pct = (row['close'] - entry_price) / entry_price * 100
            reason = []
            if row['rsi'] > 60: reason.append("RSI > 60")
            if row['macd'] < row['macd_signal']: reason.append("MACD Bear-Crossover")
            if row['close'] >= row['bb_upper']: reason.append("Kurs >= BB upper")
            if row['stoch_k'] > 80: reason.append("Stoch > 80")
            if row['close'] < entry_price * (1 - STOP_LOSS_PCT): reason.append("Stop-Loss")
            if row['close'] > entry_price * (1 + TAKE_PROFIT_PCT): reason.append("Take-Profit")
            warum = "\n".join(reason)
            was_besser = "TP/SL optimieren" if abs(pnl_pct) < 0.5 else "Timing verbessern"
            entries[-1].update({
                'exit_time': i,
                'exit_price': row['close'],
                'gewinn_verlust': pnl_pct,
                'kurzanalyse': entry_reasons,
                'warum_gewinn_verlust': warum,
                'was_besser': was_besser,
                'score': entry_score,
                'richtung': position
            })
            position = None
            trade_dir = None
        elif position == 'short' and exit_short:
            pnl_pct = (entry_price - row['close']) / entry_price * 100
            reason = []
            if row['rsi'] < 40: reason.append("RSI < 40")
            if row['macd'] > row['macd_signal']: reason.append("MACD Bull-Crossover")
            if row['close'] <= row['bb_lower']: reason.append("Kurs <= BB lower")
            if row['stoch_k'] < 20: reason.append("Stoch < 20")
            if row['close'] > entry_price * (1 + STOP_LOSS_PCT): reason.append("Stop-Loss")
            if row['close'] < entry_price * (1 - TAKE_PROFIT_PCT): reason.append("Take-Profit")
            warum = "\n".join(reason)
            was_besser = "TP/SL optimieren" if abs(pnl_pct) < 0.5 else "Timing verbessern"
            entries[-1].update({
                'exit_time': i,
                'exit_price': row['close'],
                'gewinn_verlust': pnl_pct,
                'kurzanalyse': entry_reasons,
                'warum_gewinn_verlust': warum,
                'was_besser': was_besser,
                'score': entry_score,
                'richtung': position
            })
            position = None
            trade_dir = None
        # Entry jetzt nur wenn keine offene Position
        if not position and entries_this_bar:
            for mode, curr_score, curr_reasons in entries_this_bar:
                entries.append({
                    'asset': asset,
                    'entry_time': i,
                    'entry_price': row['close'],
                    'score': curr_score,
                    'kurzanalyse': curr_reasons,
                    'richtung': mode
                })
                entry_price = row['close']
                position = mode
                entry_score = curr_score
                entry_reasons = curr_reasons
                trade_dir = mode
                break  # Nur erster Entry pro Bar
    # Noch offenen Trade schließen
    if position and entries:
        last = df.iloc[-1]
        if position == 'long':
            pnl_pct = (last['close'] - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - last['close']) / entry_price * 100
        entries[-1].update({
            'exit_time': last.name,
            'exit_price': last['close'],
            'gewinn_verlust': pnl_pct,
            'kurzanalyse': entry_reasons,
            'warum_gewinn_verlust': "Ende Zeitraum",
            'was_besser': "Früher exitten",
            'score': entry_score,
            'richtung': trade_dir
        })
    trades = pd.DataFrame(entries)
    # Zusätzliche Spalten/IDs
    trades.insert(0, 'trade_id', range(1, len(trades)+1))
    trades['kaufpreis'] = trades['entry_price']
    trades['verkaufspreis'] = trades['exit_price']
    return trades

# Optional: Haupt-Guard für Modultest
if __name__ == "__main__":
    # Kurzer Self-Test
    print("strategy.py – Self-Test – Beispielaufruf (bitte im Batch-Modul nutzen)")
