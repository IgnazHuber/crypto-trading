# d:\Projekte\crypto_trading\crypto_trading\analysis\btc-1h-1y-10kpi.py

import pandas as pd
import os
from tqdm import tqdm

from crypto_trading.config.settings import *
from crypto_trading.core.candlestick import CANDLE_FUNC
from crypto_trading.core.indicators import add_indicators
from crypto_trading.core.regimes import classify_market_regime
from crypto_trading.core.strategy import run_strategy
from crypto_trading.visualization.plot_trades import plot_strategy_chart_with_equity
from crypto_trading.reporting.export_excel import export_kpis_and_trades
from crypto_trading.visualization.compare_strategies import compare_strategies_html

def main():
    df = pd.read_parquet(PARQUET_PATH)
    if 'timestamp' not in df:
        df['timestamp'] = pd.date_range("2020-01-01", periods=len(df), freq="h")
    df = add_indicators(df)
    regimes = classify_market_regime(df)
    trades_dict = {}
    all_trades = []
    kpi_list = []
    sheet_names = []
    # Fortschrittsbalken für alle Strategien!
    for strat_name, candle_key, regime_key, regime_desc in tqdm(STRATEGY_CONFIG, desc="Strategien", ncols=80):
        candle = CANDLE_FUNC[candle_key]
        regime_mask = regimes[regime_key] if regime_key in regimes else pd.Series([False]*len(df), index=df.index)
        trades = run_strategy(df, candle, regime_mask, strat_name, regime_desc, asset=ASSET)
        all_trades.append(trades)
        trades_dict[safe_sheet_name(strat_name)] = trades
        sheet_names.append(safe_sheet_name(strat_name))
        chart_path = os.path.join(RESULTS_DIR, f"{safe_sheet_name(strat_name)}.html")
        plot_strategy_chart_with_equity(df, trades, strat_name, chart_path)
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
        kpi_list.append({
            "Strategy": strat_name,
            "Regime": regime_desc,
            "Trades": int(num_trades),
            "WinRate": round(win_rate, 1),
            "TotalPnL_abs": round(total_pnl, 1),
            "TotalPnL_pct": round(100*total_pnl/START_CAPITAL, 1) if START_CAPITAL else 0,
            "Avg PnL %": round(avg_pnl, 1),
            "Max Win %": round(max_win, 1),
            "Max Loss %": round(max_loss, 1),
            "End Capital": round(end_capital, 1)
        })
    export_kpis_and_trades(kpi_list, all_trades, OUTPUT_EXCEL, sheet_names)
    # --- Vergleichsmodul immer ausführen (z.B. erste zwei Strategien vergleichen) ---
    if len(sheet_names) >= 2:
        compare_strategies_html(df, trades_dict, sheet_names[0], sheet_names[1],
            os.path.join(RESULTS_DIR, f"vergleich_{sheet_names[0]}_vs_{sheet_names[1]}.html"))
    print("\nAlles fertig. Siehe Ergebnisse im Results-Ordner.")

if __name__ == "__main__":
    main()
