"""
multi_symbol_loop.py
Continuous batch execution loop for intraday multi-symbol trading.
"""

import argparse
import subprocess
import sys
import time

from config import TRADE_INTERVAL_SECONDS
from logger import log
from market_hours import is_market_open, market_status_str, minutes_until_open


WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]
INTERVAL_SECONDS = TRADE_INTERVAL_SECONDS


def _run_symbol(symbol: str, dry_run: bool) -> None:
    cmd = [sys.executable, "main.py", "--symbol", symbol]
    if dry_run:
        cmd.append("--dry-run")

    log(f"Processing {symbol}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            log(f"main.py exited with code {result.returncode} for {symbol}")
        if result.stderr:
            log(f"STDERR from main.py for {symbol}:\n{result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        log(f"main.py timed out after 120 seconds for {symbol} -- skipping.")
    except Exception as exc:
        log(f"Failed to launch main.py for {symbol}: {exc}")


def run_trading_loop(
    watchlist: list[str] | None = None,
    interval_seconds: int = INTERVAL_SECONDS,
    dry_run: bool = False,
    iterations: int | None = None,
) -> None:
    symbols = watchlist or WATCHLIST
    mode = "DRY-RUN" if dry_run else "LIVE"
    log(
        f"Multi-symbol trading loop starting -- symbols={','.join(symbols)}, "
        f"interval={interval_seconds}s, mode={mode}"
    )

    count = 0

    try:
        while True:
            if not is_market_open():
                wait_mins = minutes_until_open()
                log(f"Market closed. {market_status_str()} | Next open in ~{wait_mins} min.")
                for _ in range(min(wait_mins, 60)):
                    time.sleep(60)
                continue

            log(f"--- Batch {count + 1} | {market_status_str()} ---")
            for symbol in symbols:
                _run_symbol(symbol, dry_run)

            count += 1
            if iterations and count >= iterations:
                log(f"Reached {iterations} iterations. Stopping.")
                break

            log(f"Sleeping {interval_seconds}s until next cycle...")
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        log("Loop stopped by user (Ctrl+C).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-symbol trading bot loop.")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to main.py (no orders placed).")
    parser.add_argument("--iterations", type=int, default=None, help="Stop after N batches (default: run forever).")
    args = parser.parse_args()

    run_trading_loop(WATCHLIST, INTERVAL_SECONDS, args.dry_run, args.iterations)
