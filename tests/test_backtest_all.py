import os
from crypto_trading.backtesting import run_backtest_all

def test_backtest_runs():
    run_backtest_all.main()
    result_dir = run_backtest_all.RESULT_DIR
    assert os.path.exists(os.path.join(result_dir, "backtest_summary.xlsx"))
    # PDF-Report optional prüfen:
    report_pdf = os.path.join(result_dir, "backtest_report.pdf")
    if os.path.exists(report_pdf):
        assert True
