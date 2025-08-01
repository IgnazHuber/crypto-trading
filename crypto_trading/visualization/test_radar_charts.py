elif mode == "radar":
    from crypto_trading.visualization.radar_charts import plot_trade_radar
    from crypto_trading.data.fetch_yfinance import fetch_yfinance_data
    from crypto_trading.indicators.ta_standard import add_indicators
    from crypto_trading.strategy.macd_rsi_bollinger import MACD_RSI_Bollinger_Strategy
    import os

    df = fetch_yfinance_data("BTC-USD", period="6mo", interval="1d")
    df = df.set_index("date")
    df = add_indicators(df).dropna()
    trades = MACD_RSI_Bollinger_Strategy().run(df)

    if trades.empty:
        print("Keine Trades gefunden, keine Radar-Charts erstellt.")
    else:
        first_trade = trades.iloc[0].to_dict()
        plot_trade_radar(first_trade, out_dir="radar_trade_1")
