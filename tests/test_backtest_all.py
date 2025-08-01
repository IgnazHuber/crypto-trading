# tests/test_backtest_all.py
import os
from crypto_trading.backtesting import run_backtest_all

def test_backtest_runs():
    run_backtest_all.main()
    assert os.path.exists(os.path.join(run_backtest_all.RESULTS_DIR, "backtest_summary.xlsx"))
    assert os.path.exists(os.path.join(run_backtest_all.RESULTS_DIR, "backtest_report.pdf"))
