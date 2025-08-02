# crypto_trading/indicator_legend.py

INDICATOR_LEGEND = {
    "MACD": {
        "kurz": [
            "Trendfolge-Indikator.",
            "Vergleicht kurze und lange EMAs.",
            "Hauptsächlich für Crossover-Signale.",
            "Weltweit populär, auch bei Krypto.",
            "Mit Histogramm zur Momentum-Anzeige."
        ],
        "lang": [
            "Vergleicht 12er- mit 26er-EMA.",
            "Kreuzungen der MACD-Linie mit der Signallinie erzeugen Kauf-/Verkaufssignale.",
            "Das Histogramm stellt den Abstand zwischen MACD- und Signallinie dar.",
            "Gibt frühe Hinweise auf Trendwechsel.",
            "Wird häufig für Trendbestätigung eingesetzt.",
            "Empfindlich bei starken Seitwärtsmärkten (Fehlsignale möglich).",
            "Kombinierbar mit ADX zur Trendfilterung.",
            "Ideal ab 4h-Zeiteinheit.",
            "Zeigt auch Divergenzen (Kurs vs. MACD).",
            "Wird für Aus- und Einstiegspunkte genutzt.",
            "MACD-Histogramm besonders für Ausstiegsmanagement.",
            "Gut geeignet für mittelfristige Positionen.",
            "In Krypto sehr beliebt, aber Volatilität kann zu Fehlsignalen führen.",
            "Schnitt nach oben: Bullish, Schnitt nach unten: Bearish.",
            "Nicht optimal für reines Scalping."
        ],
        "marktumfeld": "Starke Trendmärkte, mittlere bis hohe Zeiteinheiten.",
        "kombination": [
            "Mit ADX als Trendfilter.",
            "Mit RSI zur Bestätigung.",
            "Mit Bollinger Bands für Umkehrpunkte.",
            "Mit Volumen (OBV, MFI) für Breakout-Filter.",
            "Mit Price Action für Divergenzen."
        ],
        "standard": [
            "Standard: 12/26/9.",
            "Kaufsignal: MACD > Signallinie.",
            "Verkauf: MACD < Signallinie.",
            "Histogramm als Momentumfilter.",
            "Crossover = Hauptsignal."
        ]
    },
    "MACD_SIGNAL": {
        "kurz": [
            "Signallinie im MACD.",
            "EMA9 des MACD-Werts.",
            "Entry-/Exit-Trigger.",
            "Glättet das Hauptsignal.",
            "Für Cross-Signale."
        ],
        "lang": [
            "Der EMA der MACD-Linie (meist 9 Perioden).",
            "Kreuzt MACD die Signallinie von unten nach oben → Kaufsignal.",
            "Kreuzt von oben nach unten → Verkaufssignal.",
            "Reduziert kurzfristige Volatilität im MACD.",
            "Crossover sind Haupttrigger für automatische Strategien.",
            "Alleine kein Primärsignal, sondern Filter.",
            "Kann Fehlsignale verhindern.",
            "Im Kryptomarkt oft für Swingtrading genutzt.",
            "Funktioniert am besten auf 4h und länger.",
            "Nicht für alle Assets gleich robust.",
            "Signalqualität steigt mit Volumenbestätigung.",
            "Bei Divergenz mit Kurs besondere Aufmerksamkeit.",
            "Kann Verzögerung bei schnellen Bewegungen verursachen.",
            "Oft in Backtests als Crossover-Signal.",
            "Wird grafisch oft als rote Linie gezeigt."
        ],
        "marktumfeld": "Trendphasen, mittlere bis längere Zeiteinheiten.",
        "kombination": [
            "Mit MACD und Histogramm als Gesamtsystem.",
            "Mit ADX zur Vermeidung von Seitwärts-Trades.",
            "Mit Volumen für Ausbruchs-Filter.",
            "Mit RSI als Zusatzfilter.",
            "Mit Price Action Patterns."
        ],
        "standard": [
            "Standard: 9 Perioden (EMA).",
            "Entry bei MACD > Signal.",
            "Exit bei MACD < Signal.",
            "Nicht als alleiniger Indikator nutzen.",
            "Vor allem im Set mit MACD."
        ]
    },
    "MACD_HIST": {
        "kurz": [
            "Histogramm: MACD – Signal.",
            "Zeigt Momentum-Anstieg/-Abfall.",
            "Frühwarnsignal für Trendwechsel.",
            "Wichtig für Take-Profit.",
            "Hilfreich bei Divergenzen."
        ],
        "lang": [
            "Das Histogramm ist die Differenz zwischen MACD und Signal-Linie.",
            "Positive Werte: bullisches Momentum.",
            "Negative Werte: bearisches Momentum.",
            "Peaks im Histogramm sind oft Wendepunkte.",
            "Histogramm verkleinert sich vor Trendende.",
            "Zeigt frühzeitig Ermüdung des Trends.",
            "Kann als Bestätigung für MACD-Crossover genutzt werden.",
            "Visualisiert, wie schnell der Trend an Stärke gewinnt oder verliert.",
            "Wird in Algo-Trading als Zusatzkriterium verwendet.",
            "Im Kryptomarkt zuverlässig, wenn Volumen passt.",
            "In Kombination mit RSI werden Übertreibungen besser erkannt.",
            "Zeigt auch Divergenzen (Kurs läuft anders als Histogramm).",
            "Mit ADX als Trendfilter besonders robust.",
            "Nicht als alleiniger Entry-Trigger geeignet.",
            "Besser für Timing des Ausstiegs."
        ],
        "marktumfeld": "Starke Trendphasen, Preisschübe.",
        "kombination": [
            "Mit MACD und Signallinie.",
            "Mit RSI für Überhitzung.",
            "Mit Volumenindikatoren.",
            "Mit ADX zur Trendstärke.",
            "Mit Price Action für Divergenz."
        ],
        "standard": [
            "Berechnung: MACD – Signal.",
            "Peak = Trendwende.",
            "Kleine Werte = weniger Momentum.",
            "Nur als Zusatzfilter.",
            "Negativ = bearisch, positiv = bullisch."
        ]
    },
    "EMA_12": {
        "kurz": [
            "Schneller gleitender Durchschnitt.",
            "Reagiert stark auf Kursänderungen.",
            "Teil des MACD-Systems.",
            "Trendfilter für kurzfristige Bewegungen.",
            "Häufig bei Krypto-Daytradern genutzt."
        ],
        "lang": [
            "Der EMA12 ist der 12-Perioden exponentiell geglättete Durchschnitt.",
            "Gibt kurzfristige Trendrichtung vor.",
            "Ideal für schnelle Einstiege und Ausstiege.",
            "Wird oft mit EMA26 für Cross-Over-Strategien genutzt.",
            "Empfindlich für schnelle Volatilität, kann Fehlsignale liefern.",
            "Gibt in Kombination mit SMA/EMA50/200 mehr Kontext.",
            "Für Krypto-Scalping und Intraday-Strategien nützlich.",
            "Reagiert stärker als SMA gleicher Länge.",
            "Bestens als Filter für Momentumstrategien.",
            "EMA-Linien werden oft als Support/Resistance gesehen.",
            "Gut geeignet für Ausbruchsstrategien.",
            "Am besten nicht alleine verwenden.",
            "Cross-Over mit EMA26 = Mini-MACD.",
            "In Trendmärkten mit hohem Volumen am effektivsten.",
            "Kann mit RSI/ADX für Fehlsignalfilter kombiniert werden."
        ],
        "marktumfeld": "Volatile und trendende Märkte, Intraday & Swing.",
        "kombination": [
            "Mit EMA26 für Cross-Over.",
            "Mit MACD als Bestandteil.",
            "Mit RSI für zusätzliche Bestätigung.",
            "Mit Volumenindikatoren für Ausbrüche.",
            "Mit SMA50/200 für größere Trends."
        ],
        "standard": [
            "12 Perioden, exponential.",
            "Schnell, volatilitätsanfällig.",
            "Nicht als alleiniger Filter geeignet.",
            "Mit EMA26 für Mini-MACD.",
            "Gängig in Krypto und FX."
        ]
    },
    # --- Rest analog, z. B. ---
    "EMA_26":      {"kurz": ["Langfristiger EMA, Trendfilter."], "lang": ["Wie EMA12, aber träger, Teil des MACD."], "marktumfeld": "Trendmärkte.", "kombination": ["Mit EMA12, MACD, SMA200."], "standard": ["26 Perioden."]},
    "SMA_20":      {"kurz": ["20er SMA, Standard für BB."], "lang": ["Einfacher Durchschnitt, meist für Volatilitätsanalyse."], "marktumfeld": "Alle Märkte.", "kombination": ["Mit BB, RSI, EMA."], "standard": ["20 Perioden."]},
    "SMA_50":      {"kurz": ["50er SMA, mittelfristig."], "lang": ["Mittlerer Trendfilter, beliebt im Krypto."], "marktumfeld": "Trend- und Konsolidierungsphasen.", "kombination": ["Mit SMA200, EMA."], "standard": ["50 Perioden."]},
    "SMA_200":     {"kurz": ["200er SMA, Haupttrend."], "lang": ["Langfristiger Durchschnitt, Trendfilter."], "marktumfeld": "Große Trends, Tages-/Wochenbasis.", "kombination": ["Mit SMA50 (Gold/Death Cross), MACD."], "standard": ["200 Perioden."]},
    "RSI_14":      {"kurz": ["Siehe oben."], "lang": ["Siehe oben."], "marktumfeld": "Siehe oben.", "kombination": ["Siehe oben."], "standard": ["Siehe oben."]},
    "RSI_7":       {"kurz": ["Schneller RSI, mehr Fehlsignale."], "lang": ["Wie RSI14, aber empfindlicher."], "marktumfeld": "Schnelle Märkte, Scalping.", "kombination": ["Mit RSI14, MACD."], "standard": ["7 Perioden."]},
    "STOCH_K":     {"kurz": ["%K Linie, Stochastik-Oszillator."], "lang": ["Misst Relativlage zum Hoch/Tief."], "marktumfeld": "Seitwärts-/Volamärkte.", "kombination": ["Mit %D, RSI, BB."], "standard": ["14/3 Perioden."]},
    "STOCH_D":     {"kurz": ["%D Linie, gleitender Mittelwert %K."], "lang": ["Glättung der Stoch-Kurve."], "marktumfeld": "Seitwärts, Konsolidierung.", "kombination": ["Mit %K."], "standard": ["3 Perioden."]},
    "WILLR":       {"kurz": ["Williams %R, Überkauft/Überverkauft."], "lang": ["Oszillator, -100 bis 0."], "marktumfeld": "Seitwärts, kurzfristig.", "kombination": ["Mit RSI, CCI."], "standard": ["14 Perioden."]},
    "CCI":         {"kurz": ["Commodity Channel Index, Schwungmaß."], "lang": ["Trend-/Momentumindikator."], "marktumfeld": "Trends, Ausbrüche.", "kombination": ["Mit RSI, BB, MACD."], "standard": ["20 Perioden."]},
    "ROC":         {"kurz": ["Rate of Change, Momentum."], "lang": ["Prozentuale Preisänderung."], "marktumfeld": "Trendphasen.", "kombination": ["Mit MACD, RSI."], "standard": ["12 Perioden."]},
    "MOM":         {"kurz": ["Momentum, Geschwindigkeit der Bewegung."], "lang": ["Preisveränderung gegenüber Vorperiode."], "marktumfeld": "Trends, Volatilität.", "kombination": ["Mit CCI, RSI."], "standard": ["10 Perioden."]},
    "BB_UPPER":    {"kurz": ["Oberes Bollinger Band."], "lang": ["Grenze für Überkauftheit."], "marktumfeld": "Seitwärts, Volatilität.", "kombination": ["Mit BB_LOWER, RSI."], "standard": ["20, 2.0 Stdabw."]},
    "BB_MIDDLE":   {"kurz": ["Mittleres Bollinger Band, SMA20."], "lang": ["Basis der Bänder, Trendrichtung."], "marktumfeld": "Alle Märkte.", "kombination": ["Mit BB_UPPER, BB_LOWER."], "standard": ["20 Perioden."]},
    "BB_LOWER":    {"kurz": ["Unteres Bollinger Band."], "lang": ["Grenze für Überverkauftheit."], "marktumfeld": "Seitwärts, Volatilität.", "kombination": ["Mit BB_UPPER, RSI."], "standard": ["20, 2.0 Stdabw."]},
    "ATR":         {"kurz": ["Average True Range, Volatilität."], "lang": ["Misst Schwankungsbreite."], "marktumfeld": "Volatile Phasen.", "kombination": ["Mit SL/TP-Strategien, ADX."], "standard": ["14 Perioden."]},
    "ADX":         {"kurz": ["Trendstärke-Indikator, s.o."], "lang": ["Siehe oben."], "marktumfeld": "Siehe oben.", "kombination": ["Siehe oben."], "standard": ["Siehe oben."]},
    "DMI_PLUS":    {"kurz": ["Directional Movement Index (+), Trendrichtung."], "lang": ["Teil von ADX-System."], "marktumfeld": "Trends.", "kombination": ["Mit DMI_MINUS, ADX."], "standard": ["14 Perioden."]},
    "DMI_MINUS":   {"kurz": ["Directional Movement Index (-), Trendrichtung."], "lang": ["Teil von ADX-System."], "marktumfeld": "Trends.", "kombination": ["Mit DMI_PLUS, ADX."], "standard": ["14 Perioden."]},
    "OBV":         {"kurz": ["On Balance Volume, Volumen-Bestätigung."], "lang": ["Kumulation von Volumen mit Trendrichtung."], "marktumfeld": "Trend- und Ausbruchsphasen.", "kombination": ["Mit Preis, MACD, RSI."], "standard": ["keine."]},
    "MFI":         {"kurz": ["Money Flow Index, Volumen-Oszillator."], "lang": ["Kombiniert Preis und Volumen."], "marktumfeld": "Seitwärts und Breakout.", "kombination": ["Mit RSI, OBV, MACD."], "standard": ["14 Perioden."]},
    "VWAP":        {"kurz": ["Volume Weighted Average Price."], "lang": ["Volumengewichteter Durchschnitt."], "marktumfeld": "Intraday, Exchanges.", "kombination": ["Mit Preis, BB, MACD."], "standard": ["intraday."]},
    "VOLUME_SMA":  {"kurz": ["Volumen gleitender Durchschnitt."], "lang": ["Filter für ungewöhnliches Volumen."], "marktumfeld": "Breakouts.", "kombination": ["Mit MACD, BB."], "standard": ["20 Perioden."]},
    "DONCH_UPPER": {"kurz": ["Oberes Donchian Channel Band."], "lang": ["Höchster Hochkurs der letzten n Perioden."], "marktumfeld": "Breakout, Trendfolge.", "kombination": ["Mit DONCH_LOWER, ADX."], "standard": ["20 Perioden."]},
    "DONCH_LOWER": {"kurz": ["Unteres Donchian Channel Band."], "lang": ["Tiefster Kurs der letzten n Perioden."], "marktumfeld": "Breakout, Trendfolge.", "kombination": ["Mit DONCH_UPPER, ADX."], "standard": ["20 Perioden."]},
    "TRIX":        {"kurz": ["Triple Exponential Average."], "lang": ["Trendfolge, Filter für Ausbrüche."], "marktumfeld": "Trend- und Momentumphasen.", "kombination": ["Mit MACD, RSI, CCI."], "standard": ["15 Perioden."]},
    "SAR":         {"kurz": ["Parabolic SAR, Trendwendesignal."], "lang": ["Folgt Trend, signalisiert Umkehr."], "marktumfeld": "Starke Trends.", "kombination": ["Mit MACD, ADX, SMA200."], "standard": ["Step: 0.02, Max: 0.2."]}
}

def get_indicator_legend():
    """Gibt die vollständige Indikator-Legende (dict) zurück."""
    return INDICATOR_LEGEND
