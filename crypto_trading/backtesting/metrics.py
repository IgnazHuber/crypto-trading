import numpy as np
import pandas as pd

def calculate_performance_metrics(trades_df: pd.DataFrame):
    """Berechnet 30 Kennzahlen für das Portfolio."""
    if trades_df.empty:
        return {"Anzahl Trades": 0}

    trades_df = trades_df.copy()
    trades_df["pnl_abs"] = trades_df["exit_price"] - trades_df["entry_price"]
    trades_df["pnl_pct"] = (trades_df["pnl_abs"] / trades_df["entry_price"]) * 100

    # === Basiswerte ===
    num_trades = len(trades_df)
    total_pnl_abs = trades_df["pnl_abs"].sum()
    total_pnl_pct = trades_df["pnl_pct"].sum()
    avg_pnl_abs = trades_df["pnl_abs"].mean()
    avg_pnl_pct = trades_df["pnl_pct"].mean()
    win_trades = trades_df[trades_df["pnl_abs"] > 0]
    loss_trades = trades_df[trades_df["pnl_abs"] <= 0]
    win_rate = len(win_trades) / num_trades * 100
    avg_win = win_trades["pnl_abs"].mean() if not win_trades.empty else 0
    avg_loss = loss_trades["pnl_abs"].mean() if not loss_trades.empty else 0
    max_win = trades_df["pnl_abs"].max()
    max_loss = trades_df["pnl_abs"].min()

    # === Equity-Kurve ===
    returns = trades_df["pnl_pct"] / 100
    equity = (1 + returns).cumprod()
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_drawdown = drawdown.min() * 100

    # === Risikokennzahlen ===
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
    downside = returns[returns < 0]
    sortino = (returns.mean() / (downside.std() + 1e-9)) * np.sqrt(252)

    # === CAGR ===
    start_date = pd.to_datetime(trades_df["entry_date"].min())
    end_date = pd.to_datetime(trades_df["exit_date"].max())
    years = max((end_date - start_date).days / 365.25, 1e-9)
    cagr = (equity.iloc[-1] ** (1 / years) - 1) * 100

    # === Monatsstatistik ===
    returns_with_index = pd.Series(
        data=returns.values,
        index=pd.to_datetime(trades_df["exit_date"])
    )
    positive_months = (returns_with_index.resample("M").sum() > 0).sum()
    negative_months = (returns_with_index.resample("M").sum() <= 0).sum()

    # === Top 5 Trades ===
    top5 = trades_df.sort_values("pnl_abs", ascending=False).head(5)
    flop5 = trades_df.sort_values("pnl_abs", ascending=True).head(5)

    # === Weitere Kennzahlen ===
    profit_factor = abs(win_trades["pnl_abs"].sum() / loss_trades["pnl_abs"].sum()) if not loss_trades.empty else np.inf
    median_pnl = trades_df["pnl_abs"].median()
    std_pnl = trades_df["pnl_abs"].std()

    # === Ergebnis zusammenstellen ===
    summary = {
        "Anzahl Trades": num_trades,
        "Gesamt PnL (abs)": round(total_pnl_abs, 2),
        "Gesamt PnL (%)": round(total_pnl_pct, 2),
        "Durchschn. PnL abs": round(avg_pnl_abs, 2),
        "Durchschn. PnL %": round(avg_pnl_pct, 2),
        "Trefferquote (%)": round(win_rate, 2),
        "Durchschn. Gewinn-Trade": round(avg_win, 2),
        "Durchschn. Verlust-Trade": round(avg_loss, 2),
        "Max. Gewinn-Trade": round(max_win, 2),
        "Max. Verlust-Trade": round(max_loss, 2),
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "CAGR (%)": round(cagr, 2),
        "Profitfaktor": round(profit_factor, 2),
        "Median PnL": round(median_pnl, 2),
        "Std PnL": round(std_pnl, 2),
        "Positive Monate": int(positive_months),
        "Negative Monate": int(negative_months),
        "Top 5 Trades": top5.to_dict(orient="records"),
        "Flop 5 Trades": flop5.to_dict(orient="records")
    }

    return summary
