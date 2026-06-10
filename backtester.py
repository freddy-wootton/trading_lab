"""
Backtesting metrics calculator for the Trading Lab.

Usage:
    from backtester import compute_metrics
    from database import get_all_trades

    df     = get_all_trades()
    report = compute_metrics(df)
    print(report)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    """
    Compute a suite of performance metrics from a DataFrame of trades.

    Args:
        trades_df: DataFrame with at minimum a 'pnl' column and optionally a
                   'timestamp' column. All other columns are ignored.

    Returns:
        dict with keys:
            total_trades  (int)
            total_pnl     (float)
            win_rate      (float, 0–100 %)
            sharpe_ratio  (float, annualised assuming 252 trading days)
            max_drawdown  (float, peak-to-trough as a positive percentage)
            equity_curve  (pd.Series of cumulative PnL, index = trade number)
    """
    # ------------------------------------------------------------------
    # Guard: return sensible defaults for an empty / unusable DataFrame
    # ------------------------------------------------------------------
    empty_result = {
        "total_trades": 0,
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "equity_curve": pd.Series(dtype=float),
    }

    if trades_df is None or trades_df.empty:
        return empty_result

    if "pnl" not in trades_df.columns:
        return empty_result

    # Drop rows where pnl is NaN
    df = trades_df.dropna(subset=["pnl"]).copy()

    if df.empty:
        return empty_result

    # Sort by timestamp if available
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    pnl_series = df["pnl"].astype(float)
    total_trades = len(pnl_series)

    # ------------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------------
    total_pnl = float(pnl_series.sum())
    win_rate = float((pnl_series > 0).sum() / total_trades * 100)

    # ------------------------------------------------------------------
    # Equity curve (cumulative PnL)
    # ------------------------------------------------------------------
    equity_curve = pnl_series.cumsum().reset_index(drop=True)

    # ------------------------------------------------------------------
    # Sharpe ratio (annualised, assume 252 trading days)
    # ------------------------------------------------------------------
    mean_pnl = pnl_series.mean()
    std_pnl = pnl_series.std()

    if std_pnl == 0 or np.isnan(std_pnl):
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float((mean_pnl / std_pnl) * np.sqrt(252))

    # ------------------------------------------------------------------
    # Maximum drawdown (peak-to-trough as a positive percentage)
    # ------------------------------------------------------------------
    cumulative = equity_curve.values
    max_drawdown = 0.0

    if len(cumulative) > 1:
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        # Express as percentage of the peak (avoid division by zero)
        peaks_nonzero = np.where(running_max != 0, running_max, np.nan)
        pct_drawdowns = drawdowns / np.abs(peaks_nonzero) * 100
        max_drawdown = float(np.nanmax(pct_drawdowns))

    return {
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "equity_curve": equity_curve,
    }


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    import random

    random.seed(42)
    sample = pd.DataFrame(
        {"pnl": [random.uniform(-50, 80) for _ in range(100)]}
    )
    result = compute_metrics(sample)
    for key, val in result.items():
        if key != "equity_curve":
            print(f"{key:>15}: {val}")
