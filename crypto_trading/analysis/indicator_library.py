import pandas as pd
import pandas_ta as ta

# ================= Grundindikatoren für Trendklassifizierung =================
def calc_ema(df, length=20):
    df[f'ema_{length}'] = ta.ema(df['close'], length=length)
    return df

def calc_sma(df, length=50):
    df[f'sma_{length}'] = ta.sma(df['close'], length=length)
    return df

def calc_rsi(df, length=14):
    df[f'rsi_{length}'] = ta.rsi(df['close'], length=length)
    return df

def calc_macd(df, fast=12, slow=26, signal=9):
    macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
    for col in macd.columns:
        df[col.lower()] = macd[col]
    return df

def calc_mfi(df, length=14):
    df[f'mfi_{length}'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=length)
    return df

def calc_atr(df, length=14):
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=length)
    return df

def calc_bbands(df, length=20, std=2.0):
    bb = ta.bbands(df['close'], length=length, std=std)
    for col in bb.columns:
        df[col.lower()] = bb[col]
    return df

def calc_trix(df, length=30):
    trix = ta.trix(df['close'], length=length)
    if isinstance(trix, pd.DataFrame):
        for col in trix.columns:
            df[col.lower()] = trix[col]
    else:
        df[f'trix_{length}'] = trix
    return df

def calc_cci(df, length=20):
    df[f'cci_{length}'] = ta.cci(df['high'], df['low'], df['close'], length=length)
    return df

def calc_stoch(df, k=14, d=3):
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=k, d=d)
    for col in stoch.columns:
        df[col.lower()] = stoch[col]
    return df

def calc_stoch_rsi(df, length=14):
    stoch_rsi = ta.stochrsi(df['close'], length=length)
    for col in stoch_rsi.columns:
        df[col.lower()] = stoch_rsi[col]
    return df

def calc_williams_r(df, length=14):
    df[f'williams_r_{length}'] = ta.willr(df['high'], df['low'], df['close'], length=length)
    return df

def calc_ultimate_osc(df):
    df['ultimate_osc'] = ta.uo(df['high'], df['low'], df['close'])
    return df

def calc_awesome_osc(df):
    df['awesome_osc'] = ta.ao(df['high'], df['low'])
    return df

def calc_roc(df, length=10):
    df[f'roc_{length}'] = ta.roc(df['close'], length=length)
    return df

def calc_momentum(df, length=10):
    df[f'mom_{length}'] = ta.mom(df['close'], length=length)
    return df

def calc_obv(df):
    df['obv'] = ta.obv(df['close'], df['volume'])
    return df

def calc_cmf(df, length=20):
    df[f'cmf_{length}'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=length)
    return df

def calc_eom(df):
    df['eom'] = ta.eom(df['high'], df['low'], df['close'], df['volume'])
    return df

def calc_tema(df, length=20):
    df[f'tema_{length}'] = ta.tema(df['close'], length=length)
    return df

def calc_dema(df, length=20):
    df[f'dema_{length}'] = ta.dema(df['close'], length=length)
    return df

def calc_hma(df, length=20):
    df[f'hma_{length}'] = ta.hma(df['close'], length=length)
    return df

def calc_wma(df, length=20):
    df[f'wma_{length}'] = ta.wma(df['close'], length=length)
    return df

def calc_ppo(df, fast=12, slow=26, signal=9):
    ppo = ta.ppo(df['close'], fast=fast, slow=slow, signal=signal)
    for col in ppo.columns:
        df[col.lower()] = ppo[col]
    return df

def calc_kc(df, length=20, mult=2):
    kc = ta.kc(df['high'], df['low'], df['close'], length=length, mult=mult)
    for col in kc.columns:
        df[col.lower()] = kc[col]
    return df

def calc_vortex(df, length=14):
    try:
        vi = ta.vortex(df['high'], df['low'], df['close'], length=length)
        for col in vi.columns:
            df[col.lower()] = vi[col]
    except Exception as e:
        print(f"[WARN] Vortex Fehler: {e}")
    return df

def calc_dmi(df, length=14):
    try:
        dmi = ta.dm(df['high'], df['low'], length=length)
        for col in dmi.columns:
            df[col.lower()] = dmi[col]
        adx = ta.adx(df['high'], df['low'], df['close'], length=length)
        df['adx'] = adx['ADX_14']
        df['dmp'] = adx['DMP_14']
        df['dmn'] = adx['DMN_14']
    except Exception as e:
        print(f"[WARN] DMI/ADX Fehler: {e}")
    return df

# ================= ALLE INDIKATOREN + TREND-SPALTEN =================
def calc_all_indicators(df, params):
    # Spaltenharmonisierung
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume"
    })

    # Trendklassifizierung relevante Indikatoren
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema200'] = ta.ema(df['close'], length=200)
    df = calc_dmi(df)
    df = calc_atr(df)
    df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()

    # Weitere Indikatoren (alle außer Donchian)
    df = calc_ema(df, params.get("ema_length", 20))
    df = calc_sma(df, params.get("sma_length", 50))
    df = calc_rsi(df, params.get("rsi_length", 14))
    df = calc_macd(df, fast=params.get("macd_fast", 12),
                   slow=params.get("macd_slow", 26),
                   signal=params.get("macd_signal", 9))
    df = calc_mfi(df)
    df = calc_bbands(df)
    df = calc_trix(df)
    df = calc_cci(df)
    df = calc_stoch(df)
    df = calc_stoch_rsi(df)
    df = calc_williams_r(df)
    df = calc_ultimate_osc(df)
    df = calc_awesome_osc(df)
    df = calc_roc(df)
    df = calc_momentum(df)
    df = calc_obv(df)
    df = calc_cmf(df)
    df = calc_eom(df)
    df = calc_tema(df)
    df = calc_dema(df)
    df = calc_hma(df)
    df = calc_wma(df)
    df = calc_ppo(df)
    df = calc_kc(df)
    df = calc_vortex(df)

    return df
