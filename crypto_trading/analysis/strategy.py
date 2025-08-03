# crypto_trading/analysis/strategy.py

from crypto_trading.analysis.score_strategy import run_score_strategy
from crypto_trading.analysis.adaptive_strategy import run_adaptive_strategy
from crypto_trading.analysis.market_regime import detect_market_regime

def run_strategy(df, config):
    if config["mode"] == "score":
        return run_score_strategy(
            df.copy(),
            config["weights"],
            config["triggers"],
            config.get("long_threshold", 1.5),
            config.get("short_threshold", -1.5)
        )
    elif config["mode"] == "adaptive":
        df = df.copy()
        df["Regime"] = detect_market_regime(df)
        return run_adaptive_strategy(
            df,
            config["regimes_weights"],
            config["regimes_triggers"],
            config["long_thresholds"],
            config["short_thresholds"]
        )
    else:
        raise ValueError(f"Unbekannter Strategy-Mode: {config['mode']}")
