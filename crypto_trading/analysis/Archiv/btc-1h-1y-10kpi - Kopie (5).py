# btc-1h-1y-10kpi - Kopie (5).py

import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === Einstellungen ===
PARQUET_PATH = "crypto_trading/data/raw/BTCUSDT_1h_1year_ccxt.parquet"
START_CAPITAL = 10_000
ASSET = "BTCUSDT"
RESULTS_DIR = r"d:\Projekte\crypto_trading\results"
OUTPUT_EXCEL = os.path.join(RESULTS_DIR, "btc_candlestick_backtest_regime.xlsx")
OUTPUT_CHART = os.path.join(RESULTS_DIR, "btc_trade_chart.png")
MIN_CAPITAL = 100

os.makedirs(RESULTS_DIR, exist_ok=True)

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

# --- Candlestick Patterns ---
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
            stop_loss = entry_price * 0.95
            take_profit = entry_price * 1.20
            time_exceeded = (df.index.get_loc(idx) - df.index.get_loc(entry_idx) > 30)
            hit_stop = (price_now <= stop_loss)
            hit_tp = (price_now >= take_profit)
            if hit_stop or hit_tp or time_exceeded:
                exit_price = price_now
                exit_time = df.loc[idx, 'timestamp'] if 'timestamp' in df else idx
                direction_mult = 1 if direction == "long" else -1
                pnl = direction_mult * (exit_price - entry_price) / entry_price * einsatz
                pnl = max(-0.05 * einsatz, min(pnl, 0.20 * einsatz))
                capital += pnl
                trades.append({
                    "Trade-ID": int(trade_id),
                    "Asset": asset,
                    "Parquet-File": os.path.basename(PARQUET_PATH),
                    "Strategy": strat_name,
                    "Regime": regime_name,
                    "Entry Time": entry_time,
                    "Exit Time": exit_time,
                    "Entry Price": round(entry_price, 1),
                    "Exit Price": round(exit_price, 1),
                    "Einsatz": round(einsatz, 1),
                    "PnL_abs": round(pnl, 1),
                    "PnL_pct": round(pnl / einsatz * 100, 1),
                    "Kapital nach Trade": round(capital, 1)
                })
                trade_id += 1
                in_position = False
    trades_df = pd.DataFrame(trades)
    return trades_df

def plot_strategy_chart_with_equity(df, trades, strategy_name, filename):
    n_trades = len(trades)
    total_pnl = trades['PnL_abs'].sum() if n_trades > 0 else 0
    total_einsatz = trades['Einsatz'].sum() if n_trades > 0 else 1
    total_pct = total_pnl / total_einsatz * 100 if n_trades > 0 else 0

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
        subplot_titles=(
            f"{strategy_name}",
            "Portfolioverlauf (Equity Curve)"
        )
    )

    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name="Kurs"
    ), row=1, col=1)

    for _, t in trades.iterrows():
        color = "green" if t["PnL_abs"] >= 0 else "red"
        fig.add_trace(go.Scatter(
            x=[t['Entry Time']], y=[t['Entry Price']],
            mode='markers+text',
            marker=dict(symbol='triangle-up', color=color, size=18, line=dict(width=2, color='black')),
            text=[f"{int(t['Trade-ID'])}"],
            textfont=dict(size=11),
            textposition='top center',
            name=f"Entry {int(t['Trade-ID'])}",
            hovertemplate=(
                f"Trade-ID: {int(t['Trade-ID'])}<br>"
                f"Krypto: {t['Asset']}<br>"
                f"Strategie: {t['Strategy']}<br>"
                f"Entry: {t['Entry Time']}<br>"
                f"Entry-Preis: {t['Entry Price']:.1f}<br>"
                f"Einsatz: {t['Einsatz']:.1f}<br>"
            ),
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[t['Exit Time']], y=[t['Exit Price']],
            mode='markers+text',
            marker=dict(symbol='triangle-down', color=color, size=18, line=dict(width=2, color='black')),
            text=[f"{int(t['Trade-ID'])}"],
            textfont=dict(size=11),
            textposition='bottom center',
            name=f"Exit {int(t['Trade-ID'])}",
            hovertemplate=(
                f"Trade-ID: {int(t['Trade-ID'])}<br>"
                f"Krypto: {t['Asset']}<br>"
                f"Strategie: {t['Strategy']}<br>"
                f"Exit: {t['Exit Time']}<br>"
                f"Exit-Preis: {t['Exit Price']:.1f}<br>"
                f"Gewinn/Verlust: {t['PnL_abs']:.1f} €<br>"
                f"Gewinn/Verlust: {t['PnL_pct']:.1f}%<br>"
                f"Kapital nach Trade: {t['Kapital nach Trade']:.1f} €<br>"
            ),
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[t['Entry Time'], t['Exit Time']],
            y=[t['Entry Price'], t['Exit Price']],
            mode='lines',
            line=dict(color=color, width=2, dash='dot'),
            hoverinfo='skip',
            showlegend=False
        ), row=1, col=1)

    # Equity-Kurve unten
    if not trades.empty:
        trades_sorted = trades.sort_values('Exit Time')
        times = [trades_sorted.iloc[0]['Entry Time']]
        capitals = [trades_sorted.iloc[0]['Kapital nach Trade'] - trades_sorted.iloc[0]['PnL_abs']]
        for i, t in trades_sorted.iterrows():
            times.append(t['Exit Time'])
            capitals.append(t['Kapital nach Trade'])
        capitals = [round(x,1) for x in capitals]
        fig.add_trace(go.Scatter(
            x=times, y=capitals,
            mode='lines+markers',
            line=dict(color="blue", width=3),
            marker=dict(size=6),
            name="Kapital",
            hovertemplate="Zeit: %{x}<br>Kapital: %{y:.1f} €",
            showlegend=False
        ), row=2, col=1)

    fig.update_yaxes(title_text="Preis", row=1, col=1, rangemode="normal")
    fig.update_yaxes(title_text="Kapital (€)", row=2, col=1, rangemode="normal")
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(
        title={
            "text": (
                f"{strategy_name}<br>"
                f"<span style='font-size:15px;color:#888;'>Trades: {n_trades}, "
                f"Gesamtgewinn/-verlust: {total_pnl:.1f} € ({total_pct:.1f} %)</span>"
            ),
            "x": 0.5, "y": 0.96,
            "xanchor": "center", "yanchor": "top"
        },
        hovermode='closest',
        height=850,
        template="plotly_white"
    )
    fig.write_html(filename)
    print(f"[plotly] Chart mit Portfolioverlauf gespeichert: {filename}")

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
    trades_dict = {}
    for i, (strat_name, candle_key, regime_key, regime_desc) in enumerate(tqdm(STRATEGY_CONFIG, desc="Strategien", ncols=80)):
        candle = CANDLE_FUNC[candle_key]
        regime_mask = regimes[regime_key] if regime_key in regimes else pd.Series([False]*len(df), index=df.index)
        trades = run_strategy(df, candle, regime_mask, strat_name, regime_desc, direction="long", asset=ASSET)
