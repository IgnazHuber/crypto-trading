# indicator_descriptions.py

INDICATOR_INFO = {
    "ema_length": {
        "title": "Exponential Moving Average (EMA)",
        "context": "Trendfolger, reagiert schneller als SMA",
        "bullets": [
            "Glättet Kursbewegungen",
            "Gewichtet neuere Kurse stärker",
            "Nützlich für Trendbestimmung",
            "Oft in Crossover-Strategien verwendet",
            "Reagiert schneller auf Richtungswechsel"
        ]
    },
    "sma_length": {
        "title": "Simple Moving Average (SMA)",
        "context": "Grundlegender Trendfilter, stabil",
        "bullets": [
            "Einfacher Durchschnitt der Kurse",
            "Reagiert langsamer als EMA",
            "Beliebt für langfristige Trendfilter",
            "Gut zur Bestimmung von Unterstützungen/Widerständen",
            "Standard in vielen Handelssystemen"
        ]
    },
    "wma_length": {
        "title": "Weighted Moving Average (WMA)",
        "context": "Trendfolger, gewichtet letzte Kurse stärker",
        "bullets": [
            "Jüngere Kurse haben höhere Gewichtung",
            "Glättet Kursverläufe effizient",
            "Gut für kurzfristige Strategien",
            "Schneller als SMA",
            "Beliebt bei Intraday-Tradern"
        ]
    },
    "tema_length": {
        "title": "Triple Exponential Moving Average (TEMA)",
        "context": "Trendfolger mit geringer Verzögerung",
        "bullets": [
            "Kombiniert mehrere EMAs",
            "Reduziert Lag gegenüber einfachem EMA",
            "Gut bei schnellen Trendwechseln",
            "Beliebt bei kurzfristigen Strategien",
            "Glatte Signale bei geringer Verzögerung"
        ]
    },
    "dema_length": {
        "title": "Double Exponential Moving Average (DEMA)",
        "context": "Trendfolger, reagiert schneller als EMA",
        "bullets": [
            "Doppelte Glättung mit weniger Verzögerung",
            "Ideal für volatile Märkte",
            "Reagiert schneller als EMA/SMA",
            "Gut für Signalgenerierung",
            "Beliebt bei algorithmischem Trading"
        ]
    },
    "hma_length": {
        "title": "Hull Moving Average (HMA)",
        "context": "Sehr schneller, geglätteter Trendfilter",
        "bullets": [
            "Glatte Signale trotz hoher Reaktionsgeschwindigkeit",
            "Weniger Lag als EMA",
            "Ideal für Swing-Trading",
            "Oft für kurzfristige Ausbrüche genutzt",
            "Kombiniert mehrere Gewichtungsansätze"
        ]
    },
    "rsi_length": {
        "title": "Relative Strength Index (RSI)",
        "context": "Momentum-Indikator, gut in Seitwärtsmärkten",
        "bullets": [
            "Zeigt überkaufte/überverkaufte Bedingungen",
            "Werte >70 oft überkauft, <30 überverkauft",
            "Kann Divergenzen aufzeigen",
            "Gut für Mean-Reversion-Strategien",
            "Beliebt für kurzfristige Swing-Trades"
        ]
    },
    "stoch_length": {
        "title": "Stochastic Oscillator",
        "context": "Momentum-Indikator, reagiert schnell",
        "bullets": [
            "Vergleicht Schlusskurs mit Preisspanne",
            "Werte >80 überkauft, <20 überverkauft",
            "Gut für kurzfristige Reversals",
            "Beliebt bei Oszillatorstrategien",
            "Zeigt Momentumwechsel frühzeitig"
        ]
    },
    "stoch_rsi_length": {
        "title": "Stochastic RSI",
        "context": "RSI im Stochastic-Format",
        "bullets": [
            "Extrem sensibel auf Preisänderungen",
            "Zeigt überkaufte/überverkaufte Zustände sehr schnell",
            "Ideal für Scalping",
            "Beliebt bei Oszillatorstrategien",
            "Liefert frühe Einstiegssignale"
        ]
    },
    "cci_length": {
        "title": "Commodity Channel Index (CCI)",
        "context": "Momentum und Zyklus-Indikator",
        "bullets": [
            "Werte >100 zeigen Überkauf an",
            "Werte < -100 zeigen Überverkauf an",
            "Gut für Zyklusanalysen",
            "Kann Divergenzen zeigen",
            "Beliebt bei Rohstoffhandel"
        ]
    },
    "macd_fast": {
        "title": "MACD (Moving Average Convergence Divergence)",
        "context": "Trendfolger + Momentum kombiniert",
        "bullets": [
            "Differenz zweier EMAs",
            "Signal-Linie als Trigger",
            "Gut für Crossover-Signale",
            "Zeigt Trendstärke",
            "Beliebt bei mittelfristigen Strategien"
        ]
    },
    "ppo": {
        "title": "Percentage Price Oscillator (PPO)",
        "context": "Ähnlich MACD, prozentual",
        "bullets": [
            "Prozentuale MACD-Variante",
            "Gut für verschiedene Asset-Preisspannen",
            "Neutral gegenüber Preisniveaus",
            "Beliebt für Portfoliovergleiche",
            "Trend- und Momentum-Indikator"
        ]
    },
    "trix_length": {
        "title": "TRIX",
        "context": "Oszillator, geglättete dreifache EMA",
        "bullets": [
            "Filtert Marktrauschen",
            "Gut für langfristige Trends",
            "Zeigt Wendepunkte",
            "Weniger empfindlich für kleine Schwankungen",
            "Gut für Positions-Trading"
        ]
    },
    "williams_r_length": {
        "title": "Williams %R",
        "context": "Oszillator, ähnlich Stochastic",
        "bullets": [
            "Überkauft/Überverkauft-Levels",
            "Gut für kurzfristige Reversals",
            "Zeigt Momentumänderungen",
            "Beliebt in Kombination mit Trendfiltern",
            "Schnelle Reaktionszeit"
        ]
    },
    "ultimate_osc": {
        "title": "Ultimate Oscillator",
        "context": "Kombiniert 3 Zeitrahmen",
        "bullets": [
            "Reduziert Fehlsignale",
            "Berücksichtigt mehrere Zeitebenen",
            "Gut bei Divergenzen",
            "Momentum-Indikator",
            "Stabilere Signale"
        ]
    },
    "awesome_osc": {
        "title": "Awesome Oscillator",
        "context": "Momentumindikator basierend auf Mittelpunkten",
        "bullets": [
            "Vergleicht kurzfristigen mit langfristigem Momentum",
            "Gut für Trendbestätigung",
            "Erkennt Richtungswechsel",
            "Einfach zu interpretieren",
            "Beliebt in Kombination mit Breakouts"
        ]
    },
    "roc_length": {
        "title": "Rate of Change (ROC)",
        "context": "Prozentuale Preisänderung",
        "bullets": [
            "Zeigt Geschwindigkeit der Preisänderung",
            "Gut für Trendfolge",
            "Empfindlich für schnelle Bewegungen",
            "Kann Überkauft/Überverkauft zeigen",
            "Oft bei Momentumstrategien genutzt"
        ]
    },
    "mom_length": {
        "title": "Momentum",
        "context": "Einfacher Momentumindikator",
        "bullets": [
            "Zeigt Stärke der Preisbewegung",
            "Einfach und robust",
            "Gut für schnelle Strategien",
            "Hilfreich in Kombination mit Trendfiltern",
            "Sehr verbreitet"
        ]
    },
    "elder_ray": {
        "title": "Elder Ray Index",
        "context": "Trendstärke und Kauf-/Verkaufsdruck",
        "bullets": [
            "Trennt bullischen und bärischen Druck",
            "Kombiniert mit Trendfiltern sehr stark",
            "Gut für mittelfristige Strategien",
            "Entwickelt von Alexander Elder",
            "Hilfreich in Kombination mit ADX"
        ]
    },
    "obv": {
        "title": "On-Balance Volume (OBV)",
        "context": "Volumenbasierter Trendindikator",
        "bullets": [
            "Addiert Volumen bei steigenden Kursen",
            "Subtrahiert bei fallenden Kursen",
            "Bestätigung von Trends",
            "Divergenzen geben frühe Signale",
            "Gut mit Preis-Trend kombiniert"
        ]
    },
    "cmf": {
        "title": "Chaikin Money Flow (CMF)",
        "context": "Volumenbasiert",
        "bullets": [
            "Misst Kapitalfluss",
            "Berücksichtigt Preis & Volumen",
            "Zeigt Kauf- und Verkaufsdruck",
            "Gut für Divergenzen",
            "Nützlich in Kombination mit OBV"
        ]
    },
    "mfi_length": {
        "title": "Money Flow Index (MFI)",
        "context": "Volumenbasiert, ähnlich RSI",
        "bullets": [
            "Werte >80 überkauft, <20 überverkauft",
            "Misst Kauf-/Verkaufsdruck",
            "Nutzt Preis und Volumen",
            "Gut für Divergenzen",
            "Ideal bei Volumenänderungen"
        ]
    },
    "adl": {
        "title": "Accumulation/Distribution Line (ADL)",
        "context": "Volumenbasiert, Trendbestätigung",
        "bullets": [
            "Zeigt Kapitalfluss in das Asset",
            "Gut für Divergenzen",
            "Trennt Akkumulation und Distribution",
            "Bestätigung von Trends",
            "Beliebt für Volumenanalyse"
        ]
    },
    "eom": {
        "title": "Ease of Movement (EoM)",
        "context": "Volumen- und Preisbewegung",
        "bullets": [
            "Misst Aufwand vs. Preisbewegung",
            "Gut bei Ausbrüchen",
            "Zeigt Markteffizienz",
            "Beliebt für kurzfristige Analysen",
            "Einfach interpretierbar"
        ]
    },
    "vwap": {
        "title": "Volume Weighted Average Price (VWAP)",
        "context": "Intraday Benchmark, volumengewichtet",
        "bullets": [
            "Zeigt durchschnittlichen Preis nach Volumen",
            "Beliebt bei institutionellen Tradern",
            "Oft als Unterstützung/Widerstand genutzt",
            "Standard im Intraday-Handel",
            "Wichtiger Benchmark für große Orders"
        ]
    },
    "force_index": {
        "title": "Force Index",
        "context": "Kombiniert Preisänderung und Volumen",
        "bullets": [
            "Misst Kauf- und Verkaufsdruck",
            "Hilfreich bei Trendbestätigung",
            "Gut für Breakouts",
            "Zeigt Stärke von Preisbewegungen",
            "Beliebt bei kurzfristigen Strategien"
        ]
    },
    "atr_length": {
        "title": "Average True Range (ATR)",
        "context": "Volatilitätsindikator",
        "bullets": [
            "Misst durchschnittliche Kursschwankung",
            "Wichtig für Stop-Loss/Positionsgrößen",
            "Nicht richtungsabhängig",
            "Erhöht in volatilen Märkten",
            "Zeigt Marktrisiken"
        ]
    },
    "bb_length": {
        "title": "Bollinger Bänder",
        "context": "Volatilitätsbasiert",
        "bullets": [
            "Drei Bänder: Mittelwert, Ober- und Unterband",
            "Erweiterung bei hoher Volatilität",
            "Kontraktion signalisiert Ausbrüche",
            "Gut für Mean-Reversion",
            "Beliebt bei Breakout-Strategien"
        ]
    },
    "kc": {
        "title": "Keltner Channel",
        "context": "Volatilitätsbasierte Kanäle",
        "bullets": [
            "Nutzt ATR statt Standardabweichung",
            "Glatter als Bollinger Bänder",
            "Gut bei Trendphasen",
            "Signalisiert Ausbrüche",
            "Beliebt bei systematischem Trading"
        ]
    },
    "donchian": {
        "title": "Donchian Channel",
        "context": "Breakout-Indikator",
        "bullets": [
            "Zeigt höchste Hochs/niedrigste Tiefs",
            "Einfacher Ausbruchindikator",
            "Gut bei Trendfolge",
            "Beliebt bei Turtle-Trading",
            "Einfach zu interpretieren"
        ]
    },
    "vortex": {
        "title": "Vortex Indicator (VI)",
        "context": "Trend- und Richtungsindikator",
        "bullets": [
            "Vergleicht Aufwärts- und Abwärtsbewegung",
            "Zeigt Trendwechsel",
            "Gut für Trendfilter",
            "Beliebt bei systematischen Strategien",
            "Leicht kombinierbar"
        ]
    },
    "dmi": {
        "title": "Directional Movement Index (DMI)",
        "context": "Trendstärke und Richtung",
        "bullets": [
            "Teil von ADX-System",
            "Zeigt Kauf-/Verkaufsdruck",
            "Hilft bei Richtungsbestimmung",
            "Nützlich für Regimefilter",
            "Kombinierbar mit ADX"
        ]
    },
    "adx": {
        "title": "Average Directional Index (ADX)",
        "context": "Misst Trendstärke",
        "bullets": [
            "Werte >25 zeigen starke Trends",
            "Trendfolger unabhängig von Richtung",
            "Hilfreich für Regimefilter",
            "Gut zur Bestimmung von Trendphasen",
            "Stabil bei langfristigen Analysen"
        ]
    }
}
