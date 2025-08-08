# meta_gridsearch_250805a.py
SKRIPT_ID = "meta_gridsearch_250805a"
"""
Gridsearch für Score-basierte Meta-Strategien (z. B. beste min_signals-Schwelle).
"""

from .multi_strategy_score_250805a import generate_score_signals
from .portfolio_backtester_250805a import run_meta_strategy
from .kpi_analyzer_250805a import calc_kpis

def gridsearch_meta(df, regimes, strategy_config, pattern_functions, min_signals_range, start_capital=10000):
    grid_results = []
    for min_signals in min_signals_range:
        print(f"[{SKRIPT_ID}] Teste min_signals={min_signals}")
        score_df = generate_score_signals(df, regimes, strategy_config, pattern_functions, min_signals=min_signals)
        trades = run_meta_strategy(df.assign(meta_long=score_df['meta_long']), 'meta_long', start_capital=start_capital)
        kpis = calc_kpis(trades, start_capital=start_capital)
        kpis['min_signals'] = min_signals
        grid_results.append(kpis)
    return grid_results
