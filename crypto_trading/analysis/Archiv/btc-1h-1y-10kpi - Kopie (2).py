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
OUTPUT_EXCEL = "btc_candlestick_backtest_regime.xlsx"
OUTPUT_CHART = "btc_trade_chart.png"
MIN_CAPITAL = 100  # ab hier stoppen wir das Trading

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

def run_strategy(df, candle_func, regime_mask, strat_name, regime_name, direction="long", start_capital=START_CAPITAL, asset=ASSET):
    trades = []
    capital = start_capital
    in_position, entry_price, entry_idx, entry_time = False, None, None, None
    trade_id = 1
    for idx in df.index:
        if capital < MIN_CAPITAL:
            break
        entry_signal = candle_func(df).loc[idx] and regime_mask.loc[idx]
        if not in_position and entry_signal:
            einsatz = min(0.10 * capital, capital)
            if einsatz < 1:
                continue
            in_position = True
            entry_price = df.loc[idx, 'close']
            entry_idx = idx
            entry_time = df.loc[idx, 'timestamp'] if 'timestamp' in df else idx
        elif in_position:
            price_now = df.loc[idx, 'close']
            stop_loss = entry_price * 0.95  # 5% Verlust
            take_profit = entry_price * 1.20  # 20% Gewinn
            time_exceeded = (df.index.get_loc(idx) - df.index.get_loc(entry_idx) > 30)
            hit_stop = (price_now <= stop_loss)
            hit_tp = (price_now >= take_profit)
            if hit_stop or hit_tp or time_exceeded:
                exit_price = price_now
                exit_time = df.loc[idx, 'timestamp'] if 'timestamp' in df else idx
                direction_mult = 1 if direction == "long" else -1
                pnl = direction_mult * (exit_price - entry_price) / entry_price * einsatz
                pnl = max(-0.05 * einsatz, min(pnl, 0.20 * einsatz))  # Begrenzung auf -5% bis +20%
                capital += pnl
                trades.append({
                    "trade_id": trade_id,
                    "asset": asset,
                    "parquet_file": os.path.basename(PARQUET_PATH),
                    "Strategy": strat_name,
                    "Regime": regime_name,
                    "Entry Time": entry_time,
                    "Exit Time": exit_time,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Einsatz": einsatz,
                    "PnL_abs": pnl,
                    "PnL_pct": pnl / einsatz * 100,
                    "Kapital nach Trade": capital
                })
                trade_id += 1
                in_position = False
    trades_df = pd.DataFrame(trades)
    return trades_df

def plot_candlestick_trades(df, trades, filename):
    plt.figure(figsize=(18,6))
    plt.plot(df['timestamp'], df['close'], label="Kurs (Close)", color='black', linewidth=1.2)
    for _, t in trades.iterrows():
        plt.scatter(t['Entry Time'], t['Entry Price'], color="green", s=40, marker="^", label="Entry" if _==0 else "")
        plt.scatter(t['Exit Time'], t['Exit Price'], color="red", s=40, marker="v", label="Exit" if _==0 else "")
        plt.annotate(str(int(t['trade_id'])), (t['Entry Time'], t['Entry Price']), xytext=(0,10), textcoords='offset points', fontsize=8, color='green', ha='center')
        plt.annotate(str(int(t['trade_id'])), (t['Exit Time'], t['Exit Price']), xytext=(0,-15), textcoords='offset points', fontsize=8, color='red', ha='center')
    plt.title(f"{ASSET} Trades (alle Strategien kombiniert)")
    plt.xlabel("Zeit")
    plt.ylabel("Preis")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig(filename)
    plt.close()

def main():
    print(f"Lade Daten aus: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    df = df.reset_index(drop=True)
    if 'timestamp' not in df:
        df['timestamp'] = pd.date_range("2020-01-01", periods=len(df), freq="h")
    df = add_indicators(df)
    print("Erkenne Marktumfeld ...")
    regimes = classify_market_regime(df)
    print("Simuliere alle Strategien ...")
    all_perf = []
    all_trade_dfs = []
    all_trades_combined = []
    for strat_name, candle_key, regime_key, regime_desc in tqdm(STRATEGY_CONFIG, desc="Strategien", ncols=80):
        candle = CANDLE_FUNC[candle_key]
        regime_mask = regimes[regime_key] if regime_key in regimes else pd.Series([False]*len(df), index=df.index)
        trades = run_strategy(df, candle, regime_mask, strat_name, regime_desc, direction="long", asset=ASSET)
        all_trade_dfs.append(trades)
        all_trades_combined.append(trades)
        # KPIs
        if not trades.empty:
            trades['Cum Capital'] = trades['Kapital nach Trade']
            total_pnl = trades['PnL_abs'].sum()
            win_rate = (trades['PnL_abs'] > 0).mean() * 100
            num_trades = len(trades)
            avg_pnl = trades['PnL_pct'].mean()
            max_win = trades['PnL_pct'].max()
            max_loss = trades['PnL_pct'].min()
            end_capital = trades['Kapital nach Trade'].iloc[-1]
        else:
            total_pnl = 0
            win_rate = 0
            num_trades = 0
            avg_pnl = 0
            max_win = 0
            max_loss = 0
            end_capital = START_CAPITAL
        all_perf.append({
            "Strategy": strat_name,
            "Regime": regime_desc,
            "Trades": num_trades,
            "WinRate": win_rate,
            "TotalPnL_abs": total_pnl,
            "TotalPnL_pct": 100*total_pnl/START_CAPITAL,
            "Avg PnL %": avg_pnl,
            "Max Win %": max_win,
            "Max Loss %": max_loss,
            "End Capital": end_capital
        })
    perf_df = pd.DataFrame(all_perf)
    print("\n=== Zusammenfassung (alle Strategien) ===")
    print(perf_df[["Strategy", "Regime", "Trades", "WinRate", "TotalPnL_abs", "End Capital"]].to_string(index=False))
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        perf_df.to_excel(writer, sheet_name="KPIs", index=False)
        all_trades_df = pd.concat(all_trades_combined, ignore_index=True)
        all_trades_df.to_excel(writer, sheet_name="Alle Trades", index=False)
        for i, trades in enumerate(all_trade_dfs):
            strat = safe_sheet_name(STRATEGY_CONFIG[i][0])
            trades.to_excel(writer, sheet_name=strat, index=False)
    print(f"\nErgebnisse exportiert nach {OUTPUT_EXCEL}.")
    plot_candlestick_trades(df, pd.concat(all_trades_combined, ignore_index=True), OUTPUT_CHART)
    print(f"Candlestickchart mit Trades gespeichert: {OUTPUT_CHART}")

if __name__ == "__main__":
    main()
