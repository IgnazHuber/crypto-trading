import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go

# === Einstellungen ===
PARQUET_PATH = "crypto_trading/data/raw/BTCUSDT_1h_1year_ccxt.parquet"
START_CAPITAL = 10_000
ASSET = "BTCUSDT"
RESULTS_DIR = "results"
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
OUTPUT_EXCEL = os.path.join(RESULTS_DIR, "walkforward_btc1h.xlsx")

# --- Zeitfenster ---
IS_MONTHS = 6
OOS_MONTHS = 1

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

# Candlestick Pattern Functions (see previous message for full definitions)
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

def run_strategy(df, candle_func, regime_mask, strat_name, regime_name, direction="long", start_capital=START_CAPITAL, asset=ASSET, parquet_file=None, trade_id_offset=0):
    trades = []
    in_position, entry_price, entry_idx, entry_time = False, None, None, None
    trade_id = 1 + trade_id_offset
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

def plot_interactive_candlestick_with_trades(df, trades, filename_html, filename_png=None, title=None):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    trades = trades.copy()
    trades['Entry Time'] = pd.to_datetime(trades['Entry Time'])
    trades['Exit Time'] = pd.to_datetime(trades['Exit Time'])

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='Kurs'
    ))

    # Entry-Marker
    gain_trades = trades[trades['PnL_abs'] > 0]
    loss_trades = trades[trades['PnL_abs'] <= 0]

    def make_marker_scatter(trades_subset, marker_symbol, color, name):
        return go.Scatter(
            x=trades_subset['Entry Time'],
            y=trades_subset['Entry Price'],
            mode='markers+text',
            marker=dict(symbol=marker_symbol, size=12, color=color),
            text=trades_subset['trade_id'].astype(str),
            textposition="top center",
            hovertemplate=(
                "Trade ID: %{text}<br>" +
                "Krypto: " + ASSET + "<br>" +
                "Strategie: %{customdata[0]}<br>" +
                "Kaufpreis: %{y:.2f}<br>" +
                "Verkaufpreis: %{customdata[1]:.2f}<br>" +
                "Gewinn/Verlust: %{customdata[2]:.2f}<extra></extra>"
            ),
            customdata=np.stack([
                trades_subset['Strategy'],
                trades_subset['Exit Price'],
                trades_subset['PnL_abs']
            ], axis=-1),
            name=name
        )

    fig.add_trace(make_marker_scatter(gain_trades, "triangle-up", "green", "Gewinn Entry"))
    fig.add_trace(make_marker_scatter(loss_trades, "triangle-down", "red", "Verlust Entry"))

    # Exit Marker (kleiner Kreis)
    fig.add_trace(go.Scatter(
        x=trades['Exit Time'],
        y=trades['Exit Price'],
        mode='markers',
        marker=dict(symbol='circle', size=8, color='blue'),
        name='Exit'
    ))

    fig.update_layout(
        title=title or "Candlestick Chart mit Trades",
        xaxis_title="Zeit",
        yaxis_title="Preis",
        legend=dict(y=0.99, x=0.01),
        hovermode='closest'
    )

    fig.write_html(filename_html)
    if filename_png:
        #fig.write_image(filename_png)

    print(f"Chart gespeichert: {filename_html}")
    if filename_png:
        print(f"PNG exportiert: {filename_png}")

def walkforward_windows(df, is_months=6, oos_months=1):
    times = pd.to_datetime(df['timestamp'])
    win_start = times.min()
    end = times.max()
    windows = []
    while True:
        is_start = win_start
        is_end = is_start + relativedelta(months=+is_months, hours=-1)
        oos_start = is_end + pd.Timedelta(hours=1)
        oos_end = oos_start + relativedelta(months=+oos_months, hours=-1)
        if oos_end > end:
            break
        windows.append((is_start, is_end, oos_start, oos_end))
        win_start = win_start + relativedelta(months=+oos_months)
    return windows

def main():
    print(f"Lade Daten aus: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    df = df.reset_index(drop=True)
    if 'timestamp' not in df:
        df['timestamp'] = pd.date_range("2020-01-01", periods=len(df), freq="h")
    df = add_indicators(df)
    windows = walkforward_windows(df, IS_MONTHS, OOS_MONTHS)
    print(f"{len(windows)} Rolling-Fenster: jeweils {IS_MONTHS}M IS, {OOS_MONTHS}M OOS")
    perf_oos = []
    all_trades_oos = []
    for i, (is_start, is_end, oos_start, oos_end) in enumerate(tqdm(windows, desc="Walkforward", ncols=80)):
        df_is = df[(df['timestamp'] >= is_start) & (df['timestamp'] <= is_end)].copy()
        df_oos = df[(df['timestamp'] >= oos_start) & (df['timestamp'] <= oos_end)].copy()
        regimes_is = classify_market_regime(df_is)
        regimes_oos = classify_market_regime(df_oos)
        # IS: Simuliere alle Strategien, wähle beste nach IS-KPI (hier: Gesamt-PnL)
        strat_kpi = []
        for strat_name, candle_key, regime_key, regime_desc in STRATEGY_CONFIG:
            candle_is = CANDLE_FUNC[candle_key]
            regime_mask_is = regimes_is[regime_key] if regime_key in regimes_is else pd.Series([False]*len(df_is), index=df_is.index)
            trades_is = run_strategy(df_is, candle_is, regime_mask_is, strat_name, regime_desc, asset=ASSET, parquet_file=PARQUET_PATH)
            total_pnl = trades_is['PnL_abs'].sum() if not trades_is.empty else 0
            strat_kpi.append((strat_name, candle_key, regime_key, regime_desc, total_pnl))
        best_strat = max(strat_kpi, key=lambda x: x[4])
        strat_name, candle_key, regime_key, regime_desc, _ = best_strat
        # OOS: Wende beste IS-Strategie im OOS an
        candle_oos = CANDLE_FUNC[candle_key]
        regime_mask_oos = regimes_oos[regime_key] if regime_key in regimes_oos else pd.Series([False]*len(df_oos), index=df_oos.index)
        trades_oos = run_strategy(
            df_oos, candle_oos, regime_mask_oos,
            strat_name, regime_desc,
            asset=ASSET,
            parquet_file=PARQUET_PATH
        )
        all_trades_oos.append(trades_oos)
        if not trades_oos.empty:
            total_pnl = trades_oos['PnL_abs'].sum()
            win_rate = (trades_oos['PnL_abs'] > 0).mean() * 100
            num_trades = len(trades_oos)
            avg_pnl = trades_oos['PnL_pct'].mean()
            max_win = trades_oos['PnL_pct'].max()
            max_loss = trades_oos['PnL_pct'].min()
            end_capital = START_CAPITAL + total_pnl
        else:
            total_pnl = 0; win_rate = 0; num_trades = 0; avg_pnl = 0; max_win = 0; max_loss = 0; end_capital = START_CAPITAL
        perf_oos.append({
            "Window": f"{oos_start:%Y-%m}–{oos_end:%Y-%m}",
            "IS-Range": f"{is_start:%Y-%m}–{is_end:%Y-%m}",
            "OOS-Range": f"{oos_start:%Y-%m}–{oos_end:%Y-%m}",
            "Best Strategy": strat_name,
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
        # Plot für jedes Fenster speichern
        html_path = os.path.join(RESULTS_DIR, f"walkforward_window_{i+1}_{oos_start.strftime('%Y%m')}.html")
        png_path = os.path.join(RESULTS_DIR, f"walkforward_window_{i+1}_{oos_start.strftime('%Y%m')}.png")
        plot_interactive_candlestick_with_trades(df_oos, trades_oos, html_path, png_path,
                                                 title=f"Walkforward OOS Window {i+1}: {oos_start.strftime('%Y-%m')}")
    perf_oos_df = pd.DataFrame(perf_oos)
    print("\n=== Rolling Out-of-Sample Ergebnisse ===")
    print(perf_oos_df[["Window", "Best Strategy", "Trades", "WinRate", "TotalPnL_abs", "End Capital"]].to_string(index=False))
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        perf_oos_df.to_excel(writer, sheet_name="OOS_Windows", index=False)
        all_oos_trades = pd.concat(all_trades_oos, ignore_index=True)
        all_oos_trades.to_excel(writer, sheet_name="OOS_Trades", index=False)
    print(f"\nErgebnisse exportiert nach {OUTPUT_EXCEL}.")

if __name__ == "__main__":
    main()
