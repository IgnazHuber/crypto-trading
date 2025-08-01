import argparse
import os
import pandas as pd
from crypto_trading.data.fetch_yfinance import fetch_yfinance_data
from crypto_trading.data.fetch_binance import fetch_binance_data
from crypto_trading.strategy.dummy_strategy import DummyStrategy
from crypto_trading.backtesting.engine import BacktestEngine
from crypto_trading.visualization.charts import plot_price_chart

def main(mode="run"):
    if mode == "fetch":
        print("Lade Daten von yfinance...")
        df_yf = fetch_yfinance_data("BTC-USD")
        print(df_yf.head())
        print("Lade Daten von Binance...")
        df_bin = fetch_binance_data("BTCUSDT")
        print(df_bin.head())

    elif mode == "backtest":
        print("Starte Dummy-Strategie...")
        strategy = DummyStrategy()
        trades = strategy.run()
        print(f"Erzeugte {len(trades)} Trades.")
        engine = BacktestEngine(trades)
        print(engine.summary())

    elif mode == "plot":
        df = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01", periods=30),
            "Close": range(30)
        }).set_index("Date")
        plot_price_chart(df, "BTC Dummy", "output_chart.png")
        print("Chart erstellt: output_chart.png")

    elif mode == "strategy":
        from crypto_trading.indicators.ta_standard import add_indicators
        from crypto_trading.strategy.macd_rsi_bollinger import MACD_RSI_Bollinger_Strategy
        print("Hole BTC-USD Daten (yfinance)...")
        df = fetch_yfinance_data("BTC-USD", period="6mo", interval="1d")
        df = df.set_index("date")
        df = add_indicators(df).dropna()
        print("Starte MACD+RSI+Bollinger Strategie...")
        trades = MACD_RSI_Bollinger_Strategy().run(df)
        print(f"{len(trades)} Trades erzeugt")
        engine = BacktestEngine(trades)
        print(engine.summary())

    elif mode == "chart":
        from crypto_trading.visualization.trade_charts import plot_trades
        from crypto_trading.indicators.ta_standard import add_indicators
        from crypto_trading.strategy.macd_rsi_bollinger import MACD_RSI_Bollinger_Strategy
        df = fetch_yfinance_data("BTC-USD", period="6mo", interval="1d")
        df = df.set_index("date")
        df = add_indicators(df).dropna()
        trades = MACD_RSI_Bollinger_Strategy().run(df)
        plot_trades(df, trades, "BTC-USD", out_path="btc_trades.png")
        print("Chart btc_trades.png erstellt")

    elif mode == "radar":
        from crypto_trading.visualization.radar_charts import plot_trade_radar
        from crypto_trading.indicators.ta_standard import add_indicators
        from crypto_trading.strategy.macd_rsi_bollinger import MACD_RSI_Bollinger_Strategy
        df = fetch_yfinance_data("BTC-USD", period="6mo", interval="1d")
        df = df.set_index("date")
        df = add_indicators(df).dropna()
        trades = MACD_RSI_Bollinger_Strategy().run(df)
        first_trade = trades.iloc[0].to_dict() if not trades.empty else {
            "symbol": "BTC-USD",
            "entry_date": df.index[0],
            "exit_date": df.index[min(5, len(df)-1)],
            "entry_price": df["Close"].iloc[0].item(),
            "exit_price": df["Close"].iloc[min(5, len(df)-1)].item(),
            "size": 1
        }
        plot_trade_radar(first_trade, out_dir="radar_trade_1")

    elif mode == "report":
        from crypto_trading.indicators.ta_standard import add_indicators
        from crypto_trading.strategy.macd_rsi_bollinger import MACD_RSI_Bollinger_Strategy
        from crypto_trading.visualization.trade_charts import plot_trades
        from crypto_trading.visualization.radar_charts import plot_trade_radar
        from crypto_trading.visualization.pdf_report import create_pdf_report
        from crypto_trading.backtesting.metrics import calculate_performance_metrics

        print("Hole BTC-USD Daten...")
        df = fetch_yfinance_data("BTC-USD", period="6mo", interval="1d")
        df = df.set_index("date")
        df = add_indicators(df).dropna()

        print("Starte Strategie...")
        trades = MACD_RSI_Bollinger_Strategy().run(df)
        first_trade = trades.iloc[0].to_dict() if not trades.empty else {
            "symbol": "BTC-USD",
            "entry_date": df.index[0],
            "exit_date": df.index[min(5, len(df)-1)],
            "entry_price": df["Close"].iloc[0].item(),
            "exit_price": df["Close"].iloc[min(5, len(df)-1)].item(),
            "size": 1
        }

        chart_dir = "report_charts"
        os.makedirs(chart_dir, exist_ok=True)
        plot_trades(df, trades, "BTC-USD", out_path=os.path.join(chart_dir, "trades.png"))
        plot_trade_radar(first_trade, out_dir=chart_dir)

        summary = calculate_performance_metrics(trades)
        charts = {
            "Trades Chart": os.path.join(chart_dir, "trades.png"),
            "Radar Raw": os.path.join(chart_dir, "radar_raw.png"),
            "Radar Norm": os.path.join(chart_dir, "radar_norm.png")
        }
        output_pdf = "crypto_report.pdf"
        create_pdf_report(output_pdf, summary, trades, charts)

    else:
        print("Unbekannter Modus. Optionen: fetch | backtest | plot | strategy | chart | radar | report")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="run",
                        help="fetch | backtest | plot | strategy | chart | radar | report")
    args = parser.parse_args()
    main(args.mode)
