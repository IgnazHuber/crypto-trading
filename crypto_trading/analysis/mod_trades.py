import numpy as np
import pandas as pd

print("[DEBUG] mod_trades.py wurde erfolgreich geladen.")

def get_first_existing_col(df, *candidates):
    """Liefert den ersten existierenden Spaltennamen aus candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Keine der Spalten gefunden: {candidates} im DataFrame mit Spalten {df.columns.tolist()}")

def generate_signals(df: pd.DataFrame, params: dict) -> np.ndarray:
    """
    Wissenschaftliche Signalgenerierung.
    LONG: EMA20 > EMA50 und MACD > 0 und RSI < 70
    SHORT: EMA20 < EMA50 und MACD < 0 und RSI > 30
    FLAT: sonst
    Akzeptiert beide Spalten-Varianten: z.B. 'ema50' UND 'ema_50'
    """
    ema_fast = params.get("ema_fast", 20)
    ema_slow = params.get("ema_slow", 50)
    # Alle sinnvollen Varianten für die wichtigsten Indikatoren
    ema_fast_col = get_first_existing_col(df, f"ema_{ema_fast}", f"ema{ema_fast}")
    ema_slow_col = get_first_existing_col(df, f"ema_{ema_slow}", f"ema{ema_slow}")

    # MACD: Standard ist "macd_12_26_9", fallback auf "macd"
    macd_key = params.get("macd_key", "macd_12_26_9")
    if macd_key not in df.columns:
        macd_key = get_first_existing_col(df, macd_key, "macd")

    # RSI: Standard ist "rsi_14", fallback auf "rsi"
    rsi_key = params.get("rsi_key", "rsi_14")
    if rsi_key not in df.columns:
        rsi_key = get_first_existing_col(df, rsi_key, "rsi")

    rsi_upper = params.get("rsi_upper", 70)
    rsi_lower = params.get("rsi_lower", 30)

    signals = np.zeros(len(df), dtype=int)

    # Long
    long_mask = (
        (df[ema_fast_col] > df[ema_slow_col]) &
        (df[macd_key] > 0) &
        (df[rsi_key] < rsi_upper)
    )
    signals[long_mask] = 1

    # Short
    short_mask = (
        (df[ema_fast_col] < df[ema_slow_col]) &
        (df[macd_key] < 0) &
        (df[rsi_key] > rsi_lower)
    )
    signals[short_mask] = -1

    return signals

def run_meta_strategy_with_indicators(
    df: pd.DataFrame,
    params: dict,
    start_capital: float = 10_000,
    max_allocation: float = 0.1,
    stop_loss: float = 0.03,
    take_profit: float = 0.2,
    trade_every: int = 1,
    **kwargs
) -> tuple:
    """
    Führt Backtest mit kontextabhängigen Parametern und Signal-Logik aus.
    Args:
        df: DataFrame mit Kursdaten und Indikatoren
        params: dict mit Parametern für Strategie & Management
        start_capital: Startkapital in EUR/USD
        max_allocation: maximaler Kapitaleinsatz pro Trade (relativ)
        stop_loss: Stop-Loss in Prozent (relativ)
        take_profit: Take-Profit in Prozent (relativ)
        trade_every: Nur jede n-te Kerze handeln (Default: 1 = jede)
        **kwargs: Weitere optionale Parameter werden ignoriert (Kompatibilität)
    Returns:
        trades_df: DataFrame mit allen Trades
        end_capital: Kapital am Ende
        max_loss: größter Einzelverlust
    """
    print("[DEBUG] Starte run_meta_strategy_with_indicators()")
    capital = start_capital
    position = 0
    entry_price = 0
    trades = []

    signals = generate_signals(df, params)
    for i in range(1, len(df)):
        if i % trade_every != 0:
            continue

        price = df['close'].iloc[i]
        signal = signals[i]
        # Einstieg Long
        if signal == 1 and position == 0:
            allocation = capital * max_allocation
            entry_price = price
            position = allocation / price
            trades.append({
                'entry_time': df['timestamp'].iloc[i],
                'entry_price': entry_price,
                'position': position,
                'direction': 'long',
                'exit_time': None,
                'exit_price': None,
                'pnl': None
            })
        # Ausstieg Long
        elif signal == 0 and position > 0:
            exit_price = price
            pnl = (exit_price - entry_price) * position
            capital += pnl
            trades[-1]['exit_time'] = df['timestamp'].iloc[i]
            trades[-1]['exit_price'] = exit_price
            trades[-1]['pnl'] = pnl
            position = 0
            entry_price = 0
        # Stop-Loss/Take-Profit prüfen
        if position > 0:
            change = (price - entry_price) / entry_price
            if change <= -stop_loss or change >= take_profit:
                exit_price = price
                pnl = (exit_price - entry_price) * position
                capital += pnl
                trades[-1]['exit_time'] = df['timestamp'].iloc[i]
                trades[-1]['exit_price'] = exit_price
                trades[-1]['pnl'] = pnl
                position = 0
                entry_price = 0

    trades_df = pd.DataFrame(trades)
    max_loss = trades_df['pnl'].min() if not trades_df.empty else 0
    end_capital = capital
    print(f"[DEBUG] Ende: Kapital={end_capital:.2f}, max_loss={max_loss:.2f}, Trades={len(trades_df)}")
    return trades_df, end_capital, max_loss

print("[DEBUG] run_meta_strategy_with_indicators ist verfügbar:", 'run_meta_strategy_with_indicators' in dir())

if __name__ == "__main__":
    print("[DEBUG] mod_trades.py als Hauptmodul gestartet.")
    # Minimaler Demo-DataFrame zum Testen (akzeptiert beide Varianten!)
    test_df = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=50, freq="D"),
        "close": np.linspace(10, 20, 50),
        "ema_20": np.linspace(10, 20, 50) + np.random.randn(50) * 0.2,
        "ema_50": np.linspace(9, 19, 50) + np.random.randn(50) * 0.2,
        "ema20": np.linspace(10, 20, 50) + np.random.randn(50) * 0.2,
        "ema50": np.linspace(9, 19, 50) + np.random.randn(50) * 0.2,
        "macd_12_26_9": np.random.randn(50),
        "macd": np.random.randn(50),
        "rsi_14": np.random.uniform(20, 80, 50),
        "rsi": np.random.uniform(20, 80, 50)
    })
    test_params = {}
    trades, end_cap, max_l = run_meta_strategy_with_indicators(test_df, test_params, trade_every=2)
    print(trades)
    print("Endkapital:", end_cap, "Maximalverlust:", max_l)
