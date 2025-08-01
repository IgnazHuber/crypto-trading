# tests/test_fetch_all.py
import os
import pytest
from crypto_trading.data import fetch_all

def test_data_fetch_runs():
    fetch_all.main()
    assert os.path.exists(os.path.join(os.path.dirname(fetch_all.__file__), "raw"))
