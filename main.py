"""
Starter script for the Trading Lab with PyTorch and SQL integration.
"""

from __future__ import annotations

# Core utilities
import argparse
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Project-specific modules
from config import API_KEY, SECRET_KEY, DEFAULT_SYMBOL, FAST_MA, SLOW_MA
from data import get_daily_bars
from logger import log
from ml_strategy import predict_signal
from database import init_db, log_trade
from execution import submit_order
from portfolio import get_account_balance
from risk import calculate_position_size


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for symbol, lookback period, and dry-run mode.
    """
    parser = argparse.ArgumentParser(description="Run the ML-based trading algorithm.")
    parser.add_argument("-s", "--symbol", default=DEFAULT_SYMBOL, help="Ticker to fetch data for.")
    parser.add_argument("-d", "--days", type=int, default=60, help="Number of days of price history to load.")
    parser.add_argument("--dry-run", action="store_true", help="Only summarize the signal, do not touch Alpaca trading APIs.")
    return parser.parse_args()

def ensure_api_keys() -> None:
    """Check that Alpaca API keys are present in the environment."""
    if not API_KEY or not SECRET_KEY:
        log("API Keys missing. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        raise RuntimeError("Missing Alpaca API Keys.")

def compute_mas(df, fast: int = FAST_MA, slow: int = SLOW_MA):
    """
    Compute Fast and Slow Moving Averages for the given dataframe.
    These are logged to the database even if the ML model uses different features.
    """
    window = df.copy()
    window["fast_ma"] = window["close"].rolling(fast).mean()
    window["slow_ma"] = window["close"].rolling(slow).mean()
    return window

def summarize(symbol: str, signal: str, latest: dict, prediction: float) -> None:
    """Print and log a summary of the current stock snapshot and ML prediction."""
    message = (
        f"{symbol} snapshot: close={latest['close']:.2f} | "
        f"prediction={prediction:.2f} | signal={signal}"
    )
    log(message)

def run_snapshot(symbol: str, days: int, dry_run: bool) -> None:
    """
    The main execution flow for a single symbol:
    1. Fetch data -> 2. Predict Signal -> 3. Log Result -> 4. Execute Trade (if not dry-run)
    """
    ensure_api_keys()
    init_db()

    # Step 1: Fetch and Prepare Data
    try:
        bars = get_daily_bars(symbol)
    except Exception as exc:
        log(f"Failed to fetch data for {symbol}: {exc}")
        raise

    if bars.empty:
        raise RuntimeError(f"No data returned for {symbol}.")

    # Reset and compute technical indicators for logging
    history = bars.tail(days).reset_index()
    history = compute_mas(history)
    latest = history.iloc[-1]
    
    # Step 2: Generate Prediction using the LSTM Model
    signal, prediction = predict_signal(history)

    # Step 3: Summarize and Log to SQL
    summarize(symbol, signal, latest, prediction)
    
    log_trade(
        symbol=symbol,
        signal=signal,
        close_price=float(latest['close']),
        fast_ma=float(latest['fast_ma']) if not hasattr(latest['fast_ma'], 'isna') or not latest['fast_ma'].isna() else None,
        slow_ma=float(latest['slow_ma']) if not hasattr(latest['slow_ma'], 'isna') or not latest['slow_ma'].isna() else None,
        prediction=prediction
    )


    if dry_run:
        log("Dry run requested; no orders will be submitted.")
        return

    # Execution phase: Submit orders based on signals
    log(f"Executing trade for {symbol} based on signal: {signal}")
    
    if signal != "flat":
        # Calculate dynamic position size based on account balance and current price
        try:
            balance = get_account_balance()
            qty = calculate_position_size(balance, float(latest['close']))
            log(f"Dynamic sizing: Balance=${balance:.2f} | Risk -> Qty={qty}")
            
            # Submit the actual order
            submit_order(symbol, "buy" if signal == "long" else "sell", qty=qty)
        except Exception as e:
            log(f"Error during execution phase: {e}")
    else:
        log(f"No action taken for signal: {signal}")

    log("Execution phase complete.")



def main() -> None:
    args = parse_args()
    run_snapshot(args.symbol.upper(), args.days, args.dry_run)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"Starter script terminated with error: {exc}")
        print(f"Starter script failed: {exc}")
        sys.exit(1)
