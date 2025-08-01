# strategy/dummy_strategy.py
import random
import pandas as pd

class DummyStrategy:
    def run(self):
        trades = []
        for i in range(5):
            trades.append({
                "symbol": "BTC-USD",
                "entry_date": f"2024-01-{i+1:02d}",
                "exit_date": f"2024-01-{i+2:02d}",
                "entry_price": 40000 + i*100,
                "exit_price": 40000 + i*100 + random.randint(-200, 300),
                "size": 1,
            })
        return pd.DataFrame(trades)
