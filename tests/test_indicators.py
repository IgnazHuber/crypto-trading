# tests/test_indicators.py

from crypto_trading.trades import generate_all_trades
from crypto_trading.indicators import INDICATORS

def test_all_indicators_present():
    trades_df, price_data = generate_all_trades(time_range="3m")
    missing = [ind for ind in INDICATORS if ind not in trades_df.columns]
    assert not missing, f"Fehlende Indikator-Spalten: {missing}"
    # Optional: Sicherstellen, dass Werte zumindest teilweise vorhanden sind
    for ind in INDICATORS:
        assert ind in trades_df.columns
        assert trades_df[ind].notna().any() or trades_df[ind].isna().all(), f"Indikator {ind} ist komplett leer!"
