# analysis/tradingmodule_250805a.py

SKRIPT_ID = "tradingmodule_250805a"

import os
import json
from .mod_data import choose_rawdata, choose_trading_frequency, load_data
from .mod_utils import add_columns_from_result
from .mod_strategy import prepare_regimes, generate_signals
from .mod_trades import run_meta_strategy_with_indicators
from .mod_plots import plot_dual_chart_with_markers

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PARAMS_PATH = os.path.join(RESULTS_DIR, "best_params.json")
with open(PARAMS_PATH, "r") as f:
    global_params = json.load(f)

def main():
    rawfile, asset, freq = choose_rawdata()
    print(f"Ausgewählt: {asset} ({freq}) [{rawfile}]")
    trade_every = choose_trading_frequency(freq)
    print(f"Tradingfrequenz: Jede {trade_every}te Candle")
    parquet_path = os.path.join("d:/Projekte/crypto_trading/crypto_trading/data/raw", rawfile)
    df = load_data(parquet_path)
    regimes = prepare_regimes(df)
    add_columns_from_result(df, regimes)
    signals = generate_signals(df, regimes, global_params)
    df['meta_long'] = signals['meta_long']
    trades = run_meta_strategy_with_indicators(
        df, 'meta_long', global_params,
        trade_every=trade_every, asset=asset,
        strategy_name=f"Meta_Score_{global_params['min_signals']}"
    )
    trades_path = os.path.join(RESULTS_DIR, f"trades_{asset}")
    trades.to_excel(trades_path + ".xlsx", index=False)
    trades.to_csv(trades_path + ".csv", index=False)
    print(f"[{SKRIPT_ID}] Alle Trades für {asset} gespeichert.")

    html_chart_path = os.path.join(RESULTS_DIR, f"{asset}_Equity_Backtest.html")
    if not trades.empty:
        plot_dual_chart_with_markers(
            df, trades, html_chart_path,
            strat_name=f"Meta_Score_{global_params['min_signals']}_{asset}", asset=asset
        )
        print(f"[{SKRIPT_ID}] Interaktives Backtest-Chart: {html_chart_path}")
    else:
        print(f"[{SKRIPT_ID}] Keine Trades, kein Chart erzeugt.")

    print(f"[{SKRIPT_ID}] Tradingmodul-Backtest abgeschlossen.")

if __name__ == "__main__":
    main()
