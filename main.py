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
from config import API_KEY, SECRET_KEY, DEFAULT_SYMBOL, FAST_MA, SLOW_MA, INTRADAY_LOOKBACK_BARS
from data import get_intraday_bars
from logger import log
from ml_strategy import predict_signal, LSTMPricePredictor, load_model
from database import close_trade, init_db, log_trade
from execution import submit_order
from portfolio import get_account_balance, get_position
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
        bars = get_intraday_bars(symbol, lookback_bars=INTRADAY_LOOKBACK_BARS)
    except Exception as exc:
        log(f"Failed to fetch data for {symbol}: {exc}")
        raise

    if bars.empty:
        raise RuntimeError(f"No data returned for {symbol}.")

    # Reset and compute technical indicators for logging
    history = bars.tail(INTRADAY_LOOKBACK_BARS).reset_index()
    history = compute_mas(history)
    latest = history.iloc[-1]
    
    # Step 2: Generate Prediction using the LSTM Model
    # Load the model once here so predict_signal doesn't reload it on every call
    device = __import__('torch').device('cuda' if __import__('torch').cuda.is_available() else 'cpu')
    model = LSTMPricePredictor(input_size=10).to(device)
    load_model(model)
    signal, prediction = predict_signal(history, symbol=symbol, model=model)

    # Step 3: Summarize and Log to SQL
    summarize(symbol, signal, latest, prediction)

    # Clean, pandas-idiomatic NaN checks for moving averages
    fast_ma = None if pd.isna(latest['fast_ma']) else float(latest['fast_ma'])
    slow_ma = None if pd.isna(latest['slow_ma']) else float(latest['slow_ma'])

    if dry_run:
        log("Dry run requested; no orders will be submitted.")
        return

    # Execution phase: Submit orders based on signals
    log(f"Executing trade for {symbol} based on signal: {signal}")
    
    if signal == "flat":
        log(f"No action taken for signal: {signal}")
    else:
        try:
            current_position_qty = get_position(symbol)
            current_price = float(latest['close'])
            log(
                f"Position check for {symbol}: current_position_qty={current_position_qty} | "
                f"intended_action={signal}"
            )

            if signal == "long":
                can_open_long = True
                if current_position_qty < 0:
                    cover_qty = abs(current_position_qty)
                    log(f"Closing short position for {symbol} at {current_price:.2f} | cover_qty={cover_qty}")
                    close_order = submit_order(symbol, "buy", qty=cover_qty)
                    if close_order:
                        closed = close_trade(symbol, current_price)
                        if closed:
                            log(
                                f"Trade closed: entry={closed.entry_price} | exit={closed.exit_price} | "
                                f"pnl={closed.pnl:.2f}"
                            )
                        else:
                            log(f"No open short trade record found to close for {symbol}.")
                    else:
                        log(f"Failed to submit cover order for {symbol}; short position left open.")
                        can_open_long = False

                if can_open_long:
                    balance = get_account_balance()
                    qty = calculate_position_size(balance, current_price)
                    log(f"Dynamic sizing: Balance=${balance:.2f} | Risk -> Buy Qty={qty}")

                    if qty == 0:
                        log("Position size too small for current balance, skipping buy.")
                    else:
                        buy_order = submit_order(symbol, "buy", qty=qty)
                        if buy_order:
                            log_trade(
                                symbol=symbol,
                                signal="long",
                                close_price=current_price,
                                fast_ma=fast_ma,
                                slow_ma=slow_ma,
                                prediction=prediction,
                                side="buy",
                                qty=qty,
                            )

            elif signal == "short":
                if current_position_qty <= 0:
                    log(
                        f"Skipping sell for {symbol}: no shares owned and short selling is disabled."
                    )
                else:
                    sell_qty = current_position_qty
                    log(
                        f"Selling owned shares for {symbol}: position_qty={current_position_qty} | "
                        f"sell_qty={sell_qty}"
                    )
                    sell_order = submit_order(symbol, "sell", qty=sell_qty)
                    if sell_order:
                        closed = close_trade(symbol, current_price)
                        if closed:
                            log(
                                f"Trade closed: entry={closed.entry_price} | exit={closed.exit_price} | "
                                f"pnl={closed.pnl:.2f}"
                            )
                        else:
                            log(f"No open long trade record found to close for {symbol}.")
                    else:
                        log(f"Failed to submit sell order for {symbol}; long position left open.")

        except Exception as e:
            log(f"Error during execution phase: {e}")

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
