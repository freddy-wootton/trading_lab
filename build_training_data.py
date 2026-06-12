"""
Fetch 2-4 weeks of intraday (5-min) data for all 5 symbols and combine into
training_data_intraday.csv.
"""

from __future__ import annotations

import pandas as pd

from data import get_intraday_bars
from logger import log


def build_multi_symbol_training_data() -> pd.DataFrame | None:
    """
    Fetch intraday bars for AAPL, MSFT, NVDA, TSLA, and GOOGL, combine them,
    and save the result to training_data_intraday.csv.
    """
    symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]
    all_data: list[pd.DataFrame] = []

    log("Building multi-symbol training dataset...")

    for symbol in symbols:
        log(f"Fetching intraday bars for {symbol}...")
        try:
            bars = get_intraday_bars(symbol, lookback_bars=1000)
            if bars.empty:
                log(f"No data returned for {symbol}")
                continue

            frame = bars.reset_index()
            if "symbol" not in frame.columns:
                frame["symbol"] = symbol
            else:
                frame["symbol"] = symbol

            all_data.append(frame)
            log(f"Fetched {len(frame)} bars for {symbol}")
        except Exception as exc:
            log(f"Failed to fetch data for {symbol}: {exc}")

    if not all_data:
        log("ERROR: No data fetched for any symbol")
        return None

    df = pd.concat(all_data, ignore_index=True)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    log(f"Combined dataset: {len(df)} rows, {df['symbol'].nunique()} unique symbols")
    if "timestamp" in df.columns and not df["timestamp"].dropna().empty:
        log(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    cols_order = [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]
    df = df[[c for c in cols_order if c in df.columns]]

    df.to_csv("training_data_intraday.csv", index=False)
    log(f"Training data saved to training_data_intraday.csv ({len(df)} rows)")

    return df


if __name__ == "__main__":
    build_multi_symbol_training_data()
