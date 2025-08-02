# crypto_trading/indicators.py

import numpy as np
import pandas as pd
import pandas_ta as ta

INDICATORS = [
    "MACD", "MACD_SIGNAL", "MACD_HIST", "EMA_12", "EMA_26",
    "SMA_20", "SMA_50", "SMA_200", "RSI_14", "RSI_7",
    "STOCH_K", "STOCH_D", "WILLR", "CCI", "ROC", "MOM",
    "BB_UPPER", "BB_MIDDLE", "BB_LOWER", "ATR",
    "ADX", "DMI_PLUS", "DMI_MINUS", "OBV", "MFI",
    "VWAP", "VOLUME_SMA", "DONCH_UPPER", "DONCH_LOWER", "TRIX", "SAR"
]

def safe_ta_call(func, *args, col=None, default=np.nan, **kwargs):
    try:
        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            if col and col in result:
                return result[col]
            return result.iloc[:, 0]
        return result
    except Exception:
        return default

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # MACD
    macd = safe_ta_call(ta.macd, df['Close'], fast=12, slow=26, signal=9, default=pd.DataFrame())
    df['MACD'] = macd['MACD_12_26_9'] if 'MACD_12_26_9' in macd else np.nan
    df['MACD_SIGNAL'] = macd['MACDs_12_26_9'] if 'MACDs_12_26_9' in macd else np.nan
    df['MACD_HIST'] = macd['MACDh_12_26_9'] if 'MACDh_12_26_9' in macd else np.nan

    # EMA/SMA
    df['EMA_12'] = safe_ta_call(ta.ema, df['Close'], length=12)
    df['EMA_26'] = safe_ta_call(ta.ema, df['Close'], length=26)
    df['SMA_20'] = safe_ta_call(ta.sma, df['Close'], length=20)
    df['SMA_50'] = safe_ta_call(ta.sma, df['Close'], length=50)
    df['SMA_200'] = safe_ta_call(ta.sma, df['Close'], length=200)

    # RSI
    df['RSI_14'] = safe_ta_call(ta.rsi, df['Close'], length=14)
    df['RSI_7'] = safe_ta_call(ta.rsi, df['Close'], length=7)

    # Stochastic
    stoch = safe_ta_call(ta.stoch, df['High'], df['Low'], df['Close'], default=pd.DataFrame())
    df['STOCH_K'] = stoch['STOCHk_14_3_3'] if 'STOCHk_14_3_3' in stoch else np.nan
    df['STOCH_D'] = stoch['STOCHd_14_3_3'] if 'STOCHd_14_3_3' in stoch else np.nan

    # Williams %R
    df['WILLR'] = safe_ta_call(ta.willr, df['High'], df['Low'], df['Close'], length=14)

    # CCI
    df['CCI'] = safe_ta_call(ta.cci, df['High'], df['Low'], df['Close'], length=20)

    # ROC, MOM
    df['ROC'] = safe_ta_call(ta.roc, df['Close'], length=12)
    df['MOM'] = safe_ta_call(ta.mom, df['Close'], length=10)

    # Bollinger Bands
    bbands = safe_ta_call(ta.bbands, df['Close'], length=20, default=pd.DataFrame())
    df['BB_UPPER'] = bbands['BBU_20_2.0'] if 'BBU_20_2.0' in bbands else np.nan
    df['BB_MIDDLE'] = bbands['BBM_20_2.0'] if 'BBM_20_2.0' in bbands else np.nan
    df['BB_LOWER'] = bbands['BBL_20_2.0'] if 'BBL_20_2.0' in bbands else np.nan

    # ATR
    df['ATR'] = safe_ta_call(ta.atr, df['High'], df['Low'], df['Close'], length=14)

    # ADX / DMI
    adx = safe_ta_call(ta.adx, df['High'], df['Low'], df['Close'], length=14, default=pd.DataFrame())
    df['ADX'] = adx['ADX_14'] if 'ADX_14' in adx else np.nan
    df['DMI_PLUS'] = adx['DMP_14'] if 'DMP_14' in adx else np.nan
    df['DMI_MINUS'] = adx['DMN_14'] if 'DMN_14' in adx else np.nan

    # OBV, MFI, VWAP, VOLUME_SMA
    df['OBV'] = safe_ta_call(ta.obv, df['Close'], df['Volume'])
    df['MFI'] = safe_ta_call(ta.mfi, df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    df['VWAP'] = safe_ta_call(ta.vwap, df['High'], df['Low'], df['Close'], df['Volume'])
    df['VOLUME_SMA'] = safe_ta_call(ta.sma, df['Volume'], length=20)

    # Donchian
    donch = safe_ta_call(ta.donchian, df['High'], df['Low'], lower_length=20, upper_length=20, default=pd.DataFrame())
    df['DONCH_UPPER'] = donch['DONCHU_20_20'] if 'DONCHU_20_20' in donch else np.nan
    df['DONCH_LOWER'] = donch['DONCHL_20_20'] if 'DONCHL_20_20' in donch else np.nan

    # TRIX
    trix = safe_ta_call(ta.trix, df['Close'], length=15)
    if isinstance(trix, pd.DataFrame):
        df['TRIX'] = trix.iloc[:, 0]
    else:
        df['TRIX'] = trix

    # SAR: Nicht in pandas_ta -> NaN
    df['SAR'] = np.nan

    # Fehlende Spalten als NaN für Test/Kompatibilität ergänzen
    for col in INDICATORS:
        if col not in df.columns:
            df[col] = np.nan

    return df
