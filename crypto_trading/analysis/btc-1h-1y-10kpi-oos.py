import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Einstellungen ===
PARQUET_PATH = "crypto_trading/data/raw/BTCUSDT_1h_1year_ccxt.parquet"
START_CAPITAL = 10_000
ASSET = "BTCUSDT"
OUTPUT_EXCEL = "btc_candlestick_oos.xlsx"
OUTPUT_CHART = "btc_equity_oos.png"

STRATEGY_CONFIG = [
    ("Bullish Engulfing + Aufwärtstrend", "bullish_engulfing", "uptrend", "EMA50>EMA200 & MACD>0"),
    ("Bearish Engulfing + Abwärtstrend", "bearish_engulfing", "downtrend", "EMA50<EMA200 & MACD<0"),
    ("Morning Star + High Volatility", "morning_star", "high_volatility", "ATR>Median"),
    ("Shooting Star + Überkauft", "shooting_star", "overbought", "RSI>70"),
    ("Hammer + Seitwärtsmarkt", "hammer", "sideways", "ADX<20"),
    ("Evening Star + BB-Upper", "evening_star", "bb_upper", "Kurs>Bollinger-Upper"),
    ("Inside Bar + Trend", "inside_bar", "trend", "Trendfilter: EMA20>EMA50>EMA200"),
    ("Doji + Überkauft/Überverkauft", "doji", "overbought_or_oversold", "RSI>80/RSI<20"),
    ("Piercing Line + Downtrend", "piercing_line", "downtrend", "EMA50<EMA200 & MACD<0"),
    ("Three White Soldiers + Breakout", "three_white_soldiers", "breakout", "BB-Width > 75%")
]

def safe_sheet_name(name, maxlen=28):
    name = re.sub(r'[\/\\\?\*\[\]\:]', '_', name)
    return name[:maxlen]

# === Candlestick-Pattern-Detection ===
def bullish_engulfing(df):
    return (
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['close'] > df['open']) &
        (df['close'] > df['open'].shift(1)) &
        (df['open'] < df['close'].shift(1))
    )

def bearish_engulfing(df):
    return (
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['close'] < df['open']) &
        (df['open'] > df['close'].shift(1)) &
        (df['close'] < df['open'].shift(1))
    )

def morning_star(df):
    return (
        (df['close'].shift(2) < df['open'].shift(2)) &
        ((df['close'].shift(1) - df['open'].shift(1)).abs() < 0.3 * (df['high'].shift(1) - df['low'].shift(1))) &
        (df['close'] > df['open']) &
        (df['close'] > ((df['open'].shift(2) + df['close'].shift(2))/2))
    )

def shooting_star(df):
    body = (df['close'] - df['open']).abs()
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
    return (upper_shadow > 2 * body) & (lower_shadow < body) & (body > 0.2 * (df['high'] - df['low']))

def hammer(df):
    body = (df['close'] - df['open']).abs()
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    return (lower_shadow > 2 * body) & (upper_shadow < body) & (body > 0.2 * (df['high'] - df['low']))

def evening_star(df):
    return (
        (df['close'].shift(2) > df['open'].shift(2)) &
        ((df['close'].shift(1) - df['open'].shift(1)).abs() < 0.3 * (df['high'].shift(1) - df['low'].shift(1))) &
        (df['close'] < df['open']) &
        (df['close'] < ((df['open'].shift(2) + df['close'].shift(2))/2))
    )

def inside_bar(df):
    return (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))

def doji(df):
    return (abs(df['close'] - df['open']) < 0.1 * (df['high'] - df['low']))

def piercing_line(df):
    return (
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['open'] < df['low'].shift(1)) &
        (df['close'] > (df['open'].shift(1) + df['close'].shift(1))/2) &
        (df['close'] < df['open'].shift(1))
    )

def three_white_soldiers(df):
    return (
        (df['close'] > df['open']) &
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['close'].shift(2) > df['open'].shift(2))
    )

CANDLE_FUNC = {
    "bullish_engulfing": bullish_engulfing,
    "bearish_engulfing": bearish_engulfing,
    "morning_star": morning_star,
    "shooting_star": shooting_star,
    "hammer": hammer,
    "evening_star": evening_star,
    "inside_bar": inside_bar,
    "doji": doji,
    "piercing_line": piercing_line,
    "three_white_soldiers": three_white_soldiers
}

def add_indicators(df):
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    df['rsi'] = df['close'].rolling(14).apply(lambda x: 100 - 100/(1 + (x.diff().clip(lower=0).mean() / abs(x.diff().clip(upper=0).mean()+1e-9))) if len(x) == 14 else np.nan)
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    df['adx'] = abs(df['high'] - df['low']).rolling(14).mean()
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    width = df['bb_upper'] - df['bb_lower']
    df['bb_width'] = width
    df['ema20'] = df['close'].ewm(span=20).mean()
    return df

def classify_market_regime(df):
    regimes = pd.DataFrame(index=df.index)
    regimes['uptrend'] = (df['ema50'] > df['ema200']) & (df['macd'] > 0)
    regimes['downtrend'] = (df['ema50'] < df['ema200']) & (df['macd'] < 0)
    regimes['sideways'] = (df['adx'] < 20)
    regimes['high_volatility'] = (df['atr'] > df['atr'].rolling(100).median())
    regimes['breakout'] = (df['bb_width'] > df['bb_width'].rolling(100).quantile(0.75))
    regimes['overbought'] = (df['rsi'] > 70)
    regimes['oversold'] = (df['rsi'] < 30)
    regimes['bb_upper'] = df['close'] > df['bb_upper']
    regimes['trend'] = (df['ema20'] > df['ema50']) & (df['ema50'] > df['ema200'])
    regimes['overbought_or_oversold'] = (df['rsi'] > 80) | (df['rsi'] < 20)
    return regimes

def run_strategy(df, candle_func, regime_mask, strat_name, regime_name, direction="long", start_capital=START_CAPITAL, asset=ASSET, parquet_file=None, offset=0):
    trades = []
    capital = start_capital
    in_position, entry_price, entry_idx, entry_time = False, None, None, None
    trade_id = 1 + offset
    for idx in df.index:
        entry_signal = candle_func(df).loc[idx] and regime_mask.loc[idx]
        if not in_position and entry_signal:
            in_position = True
            entry_price = df.loc[idx, 'close']
            entry_idx = idx
            entry_time = df.loc[idx, 'timestamp'] if 'timestamp' in df else idx
        elif in_position:
            price_now = df.loc[idx, 'close']
            if (price_now >= entry_price * 1.05) or (price_now <= entry_price * 0.97) or (df.index.get_loc(idx) - df.index.get_loc(entry_idx) > 30):
                exit_price = price_now
                exit_time = df.loc[idx, 'timestamp'] if 'timestamp' in df else idx
                pnl = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
                pct = pnl / entry_price * 100
                capital += pnl
                trades.append({
                    "trade_id": trade_id,
                    "asset": asset,
                    "parquet_file": os.path.basename(parquet_file) if parquet_file else "",
                    "Strategy": strat_name,
                    "Regime": regime_name,
                    "Entry Time": entry_time,
                    "Exit Time": exit_time,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Einsatz": start_capital,
                    "PnL_abs": pnl,
                    "PnL_pct": pct
                })
                trade_id += 1
                in_position = False
    trades_df = pd.DataFrame(trades)
    return trades_df

def plot_equity_curves(trades_in, trades_out, filename):
    plt.figure(figsize=(15,6))
    for trades, label, color in [
        (trades_in, "In-Sample", "blue"),
        (trades_out, "Out-of-Sample", "orange")
    ]:
        if not trades.empty:
            eq = trades['PnL_abs'].cumsum() + START_CAPITAL
            plt.plot(trades['Exit Time'], eq, label=label, color=color)
    plt.title("Equity Curve: In-Sample vs. Out-of-Sample")
    plt.xlabel("Zeit")
    plt.ylabel("Kapital")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def split_in_out_sample(df, split_ratio=0.7):
    split_idx = int(split_ratio * len(df))
    df_in = df.iloc[:split_idx].copy()
    df_out = df.iloc[split_idx:].copy()
    return df_in, df_out

def main():
    print(f"Lade Daten aus: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    df = df.reset_index(drop=True)
    if 'timestamp' not in df:
        df['timestamp'] = pd.date_range("2020-01-01", periods=len(df), freq="h")
    df = add_indicators(df)
    print("Splitting In-Sample/Out-of-Sample ...")
    df_in, df_out = split_in_out_sample(df, split_ratio=0.7)
    regimes_in = classify_market_regime(df_in)
    regimes_out = classify_market_regime(df_out)
    print(f"In-Sample: {df_in['timestamp'].iloc[0]} bis {df_in['timestamp'].iloc[-1]} ({len(df_in)} Zeilen)")
    print(f"Out-of-Sample: {df_out['timestamp'].iloc[0]} bis {df_out['timestamp'].iloc[-1]} ({len(df_out)} Zeilen)")
    print("Simuliere Strategien auf beiden Sets ...")
    all_perf_in, all_perf_out = [], []
    all_trades_in, all_trades_out = [], []
    for strat_name, candle_key, regime_key, regime_desc in tqdm(STRATEGY_CONFIG, desc="Strategien", ncols=80):
        # In-Sample
        candle_in = CANDLE_FUNC[candle_key]
        regime_mask_in = regimes_in[regime_key] if regime_key in regimes_in else pd.Series([False]*len(df_in), index=df_in.index)
        trades_in = run_strategy(df_in, candle_in, regime_mask_in, strat_name, regime_desc, asset=ASSET, parquet_file=PARQUET_PATH)
        all_trades_in.append(trades_in)
        # Out-of-Sample
        candle_out = CANDLE_FUNC[candle_key]
        regime_mask_out = regimes_out[regime_key] if regime_key in regimes_out else pd.Series([False]*len(df_out), index=df_out.index)
        trades_out = run_strategy(df_out, candle_out, regime_mask_out, strat_name, regime_desc, asset=ASSET, parquet_file=PARQUET_PATH, offset=len(trades_in))
        all_trades_out.append(trades_out)
        # KPIs In
        if not trades_in.empty:
            total_pnl = trades_in['PnL_abs'].sum()
            win_rate = (trades_in['PnL_abs'] > 0).mean() * 100
            num_trades = len(trades_in)
            avg_pnl = trades_in['PnL_pct'].mean()
            max_win = trades_in['PnL_pct'].max()
            max_loss = trades_in['PnL_pct'].min()
            end_capital = START_CAPITAL + total_pnl
        else:
            total_pnl = 0; win_rate = 0; num_trades = 0; avg_pnl = 0; max_win = 0; max_loss = 0; end_capital = START_CAPITAL
        all_perf_in.append({
            "Strategy": strat_name, "Regime": regime_desc,
            "Trades": num_trades, "WinRate": win_rate,
            "TotalPnL_abs": total_pnl, "TotalPnL_pct": 100*total_pnl/START_CAPITAL,
            "Avg PnL %": avg_pnl, "Max Win %": max_win, "Max Loss %": max_loss,
            "End Capital": end_capital, "Set": "In-Sample"
        })
        # KPIs Out
        if not trades_out.empty:
            total_pnl = trades_out['PnL_abs'].sum()
            win_rate = (trades_out['PnL_abs'] > 0).mean() * 100
            num_trades = len(trades_out)
            avg_pnl = trades_out['PnL_pct'].mean()
            max_win = trades_out['PnL_pct'].max()
            max_loss = trades_out['PnL_pct'].min()
            end_capital = START_CAPITAL + total_pnl
        else:
            total_pnl = 0; win_rate = 0; num_trades = 0; avg_pnl = 0; max_win = 0; max_loss = 0; end_capital = START_CAPITAL
        all_perf_out.append({
            "Strategy": strat_name, "Regime": regime_desc,
            "Trades": num_trades, "WinRate": win_rate,
            "TotalPnL_abs": total_pnl, "TotalPnL_pct": 100*total_pnl/START_CAPITAL,
            "Avg PnL %": avg_pnl, "Max Win %": max_win, "Max Loss %": max_loss,
            "End Capital": end_capital, "Set": "Out-of-Sample"
        })
    perf_df = pd.DataFrame(all_perf_in + all_perf_out)
    print("\n=== In-Sample vs. Out-of-Sample ===")
    print(perf_df[["Strategy","Set","Trades","WinRate","TotalPnL_abs","End Capital"]].to_string(index=False))
    # Excel-Export
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        perf_df.to_excel(writer, sheet_name="KPIs", index=False)
        # Tradesheets
        pd.concat(all_trades_in, ignore_index=True).to_excel(writer, sheet_name="Trades In-Sample", index=False)
        pd.concat(all_trades_out, ignore_index=True).to_excel(writer, sheet_name="Trades Out-of-Sample", index=False)
        # Einzel-Sheets pro Strategie (optional, kommentierbar)
        for i, (trades_in, trades_out) in enumerate(zip(all_trades_in, all_trades_out)):
            strat = safe_sheet_name(STRATEGY_CONFIG[i][0])
            trades_in.to_excel(writer, sheet_name=f"{strat}_IN", index=False)
            trades_out.to_excel(writer, sheet_name=f"{strat}_OUT", index=False)
    print(f"\nErgebnisse exportiert nach {OUTPUT_EXCEL}.")
    # Chart
    plot_equity_curves(pd.concat(all_trades_in, ignore_index=True), pd.concat(all_trades_out, ignore_index=True), OUTPUT_CHART)
    print(f"Equity-Chart gespeichert: {OUTPUT_CHART}")

if __name__ == "__main__":
    main()
