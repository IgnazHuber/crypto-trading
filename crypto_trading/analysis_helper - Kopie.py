# crypto_trading/analysis_helper.py

import numpy as np

# Literaturbasierte Bewertungs-Tabellen
# (Bsp.: Murphy 1999, Schwager, Elder, Pring, eigene Praxiserfahrung)

# Indikator- vs. Marktumfeld-Scoring (0-10: 10=perfekt)
INDICATOR_MARKTUMFELD_SCORE = {
    # (Indikator, Marktumfeld): Score
    ("MACD", "Trend"): 9,
    ("MACD", "Seitwärts"): 4,
    ("RSI", "Trend"): 6,
    ("RSI", "Seitwärts"): 9,
    ("ADX", "Trend"): 10,
    ("ADX", "Seitwärts"): 3,
    ("Stochastic", "Trend"): 5,
    ("Stochastic", "Seitwärts"): 9,
    ("Bollinger", "Volatilität"): 8,
    ("Bollinger", "Ruhig"): 6,
    # ... nach Bedarf weitere Kombinationen
}

# Kombinationen-Score (bekannt aus Literatur, eigene Erfahrung)
INDICATOR_COMBO_SCORE = {
    # tuple(sorted([ind1, ind2])): score
    ("MACD", "RSI"): 8,
    ("MACD", "ADX"): 9,
    ("MACD", "Bollinger"): 8,
    ("RSI", "Stochastic"): 8,
    ("RSI", "Bollinger"): 9,
    ("ADX", "Bollinger"): 7,
    # ... nach Bedarf weitere
}

def get_market_regime(trade):
    # Einfache Regel, ausbaubar (z.B. basierend auf ADX, ATR oder Volatilität)
    adx = trade.get("ADX", None)
    if adx is not None:
        if adx >= 25:
            return "Trend"
        else:
            return "Seitwärts"
    # Fallback
    return "Unbekannt"

def get_volatility_regime(trade):
    atr = trade.get("ATR", None)
    if atr is not None:
        if atr > 0.03:  # Beispiel-Schwelle
            return "Volatilität"
        else:
            return "Ruhig"
    return "Unbekannt"

def get_indicator_scores(trade):
    """Scoring pro Indikator vs. Marktumfeld."""
    scores = []
    marktumfeld = get_market_regime(trade)
    volatility = get_volatility_regime(trade)
    for ind in ["MACD", "RSI", "ADX", "Stochastic", "Bollinger"]:
        # Prüfe auf Relevanz im Trade
        if ind in trade:
            # Kombiniere Marktumfeld & Volatilität je nach Indikator
            context = volatility if ind == "Bollinger" else marktumfeld
            score = INDICATOR_MARKTUMFELD_SCORE.get((ind, context), "-")
            scores.append((ind, context, score))
    return scores

def get_combo_scores(trade):
    """Scoring für Indikator-Kombinationen."""
    combos = []
    used_inds = [ind for ind in ["MACD", "RSI", "ADX", "Bollinger", "Stochastic"] if ind in trade]
    for i in range(len(used_inds)):
        for j in range(i+1, len(used_inds)):
            key = tuple(sorted([used_inds[i], used_inds[j]]))
            score = INDICATOR_COMBO_SCORE.get(key, "-")
            combos.append((used_inds[i], used_inds[j], score))
    return combos

def bulletify(lines):
    """Wandelt eine Liste von Strings in Bulletpoints mit Zeilenumbruch um."""
    return "\n".join([f"• {line}" for line in lines if line])

def generate_trade_analysis(trade):
    """Erstellt kurze und lange Analyse pro Trade inkl. Bulletpoints und Scores."""
    short_lines = []
    long_lines = []

    pnl = trade.get("pnl_abs", 0)
    pnl_pct = trade.get("pnl_pct", 0)
    entry_price = trade.get("entry_price", None)
    exit_price = trade.get("exit_price", None)
    symbol = trade.get("symbol", "")
    rsi = trade.get("RSI_14", None) or trade.get("RSI", None)
    macd = trade.get("MACD", None)
    adx = trade.get("ADX", None)
    boll = trade.get("Bollinger", None) or trade.get("BB_MIDDLE", None)
    stoch = trade.get("Stochastic", None) or trade.get("STOCH", None)

    # Kurzanalyse (bulletpoints)
    short_lines.append(f"Asset: {symbol}")
    if pnl > 0:
        short_lines.append(f"Gewinn-Trade (+{pnl:.1f}) ✔️")
    elif pnl < 0:
        short_lines.append(f"Verlust-Trade ({pnl:.1f}) ❌")
    else:
        short_lines.append("Break-Even")
    if entry_price and exit_price:
        short_lines.append(f"Einstieg: {entry_price:.2f}, Ausstieg: {exit_price:.2f}")
    if rsi is not None:
        if rsi > 70:
            short_lines.append("RSI überkauft (>70)")
        elif rsi < 30:
            short_lines.append("RSI überverkauft (<30)")
        else:
            short_lines.append(f"RSI: {rsi:.1f}")
    if macd is not None:
        short_lines.append(f"MACD: {macd:.2f}")
    if adx is not None:
        short_lines.append(f"ADX: {adx:.2f}")
    if boll is not None:
        short_lines.append(f"Bollinger: {boll:.2f}")
    if stoch is not None:
        short_lines.append(f"Stochastic: {stoch:.2f}")

    # Indikator vs. Marktumfeld Scoring
    ind_scores = get_indicator_scores(trade)
    for ind, ctx, score in ind_scores:
        if score != "-":
            short_lines.append(f"{ind} im Umfeld '{ctx}': Score {score}/10")

    # Kombinationsscore
    combo_scores = get_combo_scores(trade)
    for ind1, ind2, score in combo_scores:
        if score != "-":
            short_lines.append(f"Kombi {ind1}+{ind2}: {score}/10")

    # Langanalyse (bulletpoints & literaturbasiert)
    long_lines.append(f"Trade für {symbol}: {'Profitabel' if pnl>0 else 'Verlust' if pnl<0 else 'Break-Even'}")
    if pnl != 0:
        long_lines.append(f"Absoluter {'Gewinn' if pnl>0 else 'Verlust'}: {abs(pnl):.2f}")
    if pnl_pct:
        long_lines.append(f"Prozentual: {pnl_pct:.2f} %")

    # Indikatorbasiert
    if rsi is not None:
        long_lines.append(f"RSI zum Entry: {rsi:.1f} → {'Überkauft' if rsi>70 else 'Überverkauft' if rsi<30 else 'neutral'} laut J. Welles Wilder.")
    if macd is not None:
        long_lines.append(f"MACD (Gerald Appel): {macd:.2f} → {'bullischer' if macd>0 else 'bärischer'} Impuls.")
    if adx is not None:
        long_lines.append(f"ADX: {adx:.1f} → {'starker Trend' if adx>25 else 'seitwärts'} (Standard nach Wilder).")
    if boll is not None:
        long_lines.append(f"Bollinger Band (John Bollinger): Wert {boll:.2f}.")

    # Kombinations-Score (mit Quellen)
    for ind, ctx, score in ind_scores:
        if score != "-":
            long_lines.append(f"Literatur: {ind} ist im Marktumfeld '{ctx}' mit Score {score}/10 bewertet (vgl. Murphy, Schwager).")
    for ind1, ind2, score in combo_scores:
        if score != "-":
            long_lines.append(f"Kombination {ind1}+{ind2} gilt als robust (Score {score}/10, vgl. Murphy).")

    # Beispiel für Risikomanagement und Strategieumfeld
    long_lines.append("Strategie: Kombination von Indikatoren für mehrfache Signalbestätigung – empfohlen laut Elder, Murphy.")
    long_lines.append("Risikomanagement: Fester Stop-Loss/Take-Profit laut Tradingplan.")

    # Zeilenumbrüche + Bulletpoints
    return bulletify(short_lines), bulletify(long_lines)
