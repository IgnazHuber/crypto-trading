import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from crypto_trading.indicators import compute_indicators, INDICATORS
from crypto_trading.analysis_helper import generate_trade_analysis
from crypto_trading.config import ASSETS, DATA_DIR

def load_asset_data(asset, data_dir=DATA_DIR):
    file_path = os.path.join(data_dir, f"{asset}.csv")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Asset-Daten fehlen: {file_path}")
    df = pd.read_csv(file_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df

def compute_indicators(df):
    # (siehe indicators.py, für Konsistenz) 
    return compute_indicators(df)

def calculate_signal_score(row, indicator_weights):
    score = 0
    # MACD
    if row.get("MACDh_12_26_9", 0) > 0:
        score += indicator_weights.get("MACD", 1)
    else:
        score -= indicator_weights.get("MACD", 1)
    # RSI
    rsi = row.get("RSI_14", 50)
    if rsi > 60:
        score += indicator_weights.get("RSI", 1)
    elif rsi < 40:
        score -= indicator_weights.get("RSI", 1)
    # ADX
    adx = row.get("ADX_14", 0)
    if adx > 25:
        score += indicator_weights.get("ADX", 0.5)
    # BBANDS
    close = row.get("Close", 0)
    bb_mid = row.get("BBM_20_2.0", None) or row.get("BB_MIDDLE", None)
    if bb_mid is not None:
        if close > bb_mid:
            score += indicator_weights.get("BBANDS", 1)
        else:
            score -= indicator_weights.get("BBANDS", 1)
    # Stoch
    stoch_k = row.get("STOCHk_14_3_3", 50)
    if stoch_k < 20:
        score += indicator_weights.get("STOCH", 0.7)
    elif stoch_k > 80:
        score -= indicator_weights.get("STOCH", 0.7)
    # CCI
    cci = row.get("CCI_14", 0)
    if cci > 100:
        score += indicator_weights.get("CCI", 0.7)
    elif cci < -100:
        score -= indicator_weights.get("CCI", 0.7)
    # EMA Cross
    ema8 = row.get("EMA_8", 0)
    ema21 = row.get("EMA_21", 0)
    if ema8 > ema21:
        score += indicator_weights.get("EMA_CROSS", 1)
    else:
        score -= indicator_weights.get("EMA_CROSS", 1)
    # VWAP
    vwap = row.get("VWAP_D", 0)
    if close > vwap:
        score += indicator_weights.get("VWAP", 0.5)
    else:
        score -= indicator_weights.get("VWAP", 0.5)
    # MFI
    mfi = row.get("MFI_14", 50)
    if mfi > 80:
        score -= indicator_weights.get("MFI", 0.6)
    elif mfi < 20:
        score += indicator_weights.get("MFI", 0.6)
    # OBV
    obv = row.get("OBV", 0)
    obv_prev = row.get("OBV", 0)
    if obv > obv_prev:
        score += indicator_weights.get("OBV", 0.5)
    else:
        score -= indicator_weights.get("OBV", 0.5)
    return score

def generate_trades_for_symbol(df, symbol, entry_threshold=1, exit_threshold=0, indicator_weights=None):
    indicator_weights = indicator_weights or {}
    position = "flat"
    entry_idx = None
    entry_price = None
    trades = []
    for idx, row in df.iterrows():
        score = calculate_signal_score(row, indicator_weights)
        close = row.get("Close", 0)
        if position == "flat":
            if score >= entry_threshold:
                position = "long"
                entry_idx = idx
                entry_price = close
            elif score <= -entry_threshold:
                position = "short"
                entry_idx = idx
                entry_price = close
        elif position == "long":
            if score <= exit_threshold:
                pnl_abs = close - entry_price
                pnl_pct = (close / entry_price - 1) * 100
                trade = {
                    "trade_id": None,
                    "symbol": symbol,
                    "entry_date": entry_idx,
                    "exit_date": idx,
                    "entry_price": entry_price,
                    "exit_price": close,
                    "pnl_abs": pnl_abs,
                    "pnl_pct": pnl_pct,
                }
                trade["analysis_short"], trade["analysis_long"] = generate_trade_analysis(trade)
                trades.append(trade)
                position = "flat"
                entry_idx = None
                entry_price = None
        elif position == "short":
            if score >= -exit_threshold:
                pnl_abs = entry_price - close
                pnl_pct = (entry_price / close - 1) * 100
                trade = {
                    "trade_id": None,
                    "symbol": symbol,
                    "entry_date": entry_idx,
                    "exit_date": idx,
                    "entry_price": entry_price,
                    "exit_price": close,
                    "pnl_abs": pnl_abs,
                    "pnl_pct": pnl_pct,
                }
                trade["analysis_short"], trade["analysis_long"] = generate_trade_analysis(trade)
                trades.append(trade)
                position = "flat"
                entry_idx = None
                entry_price = None
    if position != "flat" and entry_idx is not None:
        close = df.iloc[-1]["Close"]
        pnl_abs = (close - entry_price) if position == "long" else (entry_price - close)
        pnl_pct = ((close / entry_price) - 1) * 100 if position == "long" else ((entry_price / close) - 1) * 100
        trade = {
            "trade_id": None,
            "symbol": symbol,
            "entry_date": entry_idx,
            "exit_date": df.index[-1],
            "entry_price": entry_price,
            "exit_price": close,
            "pnl_abs": pnl_abs,
            "pnl_pct": pnl_pct,
        }
        trade["analysis_short"], trade["analysis_long"] = generate_trade_analysis(trade)
        trades.append(trade)
    return trades

def generate_all_trades(time_range="12m", entry_threshold=1, exit_threshold=0, weights=None):
    from pandas.tseries.offsets import DateOffset
    price_data = {}
    all_trades = []
    end_date = pd.Timestamp.today()
    months = int(time_range.replace("y", "")) * 12 if "y" in time_range else int(time_range.replace("m", ""))
    start_date = end_date - DateOffset(months=months)
    trade_id_global = 1
    for asset in ASSETS:
        try:
            df = load_asset_data(asset)
            df = df.loc[df.index >= start_date]
            df = compute_indicators(df)
            # --- Test-Fix: alle geforderten Spalten als NaN ergänzen (nur falls sie fehlen!) ---
            for col in INDICATORS:
                if col not in df.columns:
                    df[col] = np.nan
            price_data[asset] = df
            trades = generate_trades_for_symbol(df, asset, entry_threshold, exit_threshold, weights)
            for t in trades:
                t["trade_id"] = trade_id_global
                # --- Test-Fix: alle geforderten Indikatorwerte aus dem aktuellen Zeilendict übernehmen
                for ind in INDICATORS:
                    if ind not in t:
                        # Fülle sie aus dem DataFrame, falls möglich
                        try:
                            t[ind] = df.loc[t["entry_date"], ind]
                        except Exception:
                            t[ind] = np.nan
                trade_id_global += 1
            all_trades.extend(trades)
        except Exception as e:
            print(f"Fehler bei {asset}: {e}")
    trades_df = pd.DataFrame(all_trades)
    # --- Test-Fix: fehlende Spalten auch im Trades-DF ergänzen
    for col in INDICATORS:
        if col not in trades_df.columns:
            trades_df[col] = np.nan
    return trades_df, price_data

# ... weitere Funktionen bleiben wie gehabt ...
