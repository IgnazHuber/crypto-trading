import numpy as np

def classify_trend(df):
    """
    Trendklassifizierung:
    - uptrend: EMA50 > EMA200 & ADX > 20
    - downtrend: EMA50 < EMA200 & ADX > 20
    - sideways: sonst
    Erwartet: Spalten ema50, ema200, adx
    """
    if not {"ema50", "ema200", "adx"}.issubset(df.columns):
        raise ValueError("Fehlende Spalten für Trendklassifizierung")

    conditions = [
        (df["ema50"] > df["ema200"]) & (df["adx"] > 20),
        (df["ema50"] < df["ema200"]) & (df["adx"] > 20)
    ]
    choices = ["uptrend", "downtrend"]
    df["trend_class"] = np.select(conditions, choices, default="sideways")
    return df


def classify_volatility(df):
    """
    Klassifiziert Volatilität:
    - high_volatility: ATR > ATR_SMA * 1.5
    - low_volatility: ATR < ATR_SMA * 0.5
    - normal_volatility: sonst
    Erwartet: Spalte atr
    """
    if "atr" not in df.columns:
        raise ValueError("Fehlende Spalte 'atr' für Volatilitätsklassifizierung")

    df["atr_sma"] = df["atr"].rolling(14, min_periods=1).mean()
    conditions = [
        df["atr"] > df["atr_sma"] * 1.5,
        df["atr"] < df["atr_sma"] * 0.5
    ]
    choices = ["high_volatility", "low_volatility"]
    df["volatility_class"] = np.select(conditions, choices, default="normal_volatility")
    return df


def classify_volume(df):
    """
    Klassifiziert Volumen:
    - high_volume: Volume > Volume-SMA
    - low_volume: sonst
    Erwartet: Spalte volume_sma
    """
    if "volume_sma" not in df.columns:
        raise ValueError("Fehlende Spalte 'volume_sma' für Volumenklassifizierung")

    df["volume_class"] = np.where(df["volume"] > df["volume_sma"],
                                  "high_volume", "low_volume")
    return df


def apply_context_to_indicators(df):
    """
    Erstellt kombinierten Marktkontext:
    Beispiel: uptrend_high_volatility_high_volume
    """
    if not {"trend_class", "volatility_class", "volume_class"}.issubset(df.columns):
        raise ValueError("Fehlende Spalten für Marktumfeld-Zuordnung")

    df["market_context"] = (df["trend_class"].astype(str) + "_" +
                            df["volatility_class"].astype(str) + "_" +
                            df["volume_class"].astype(str))
    return df
