"""
run_loop.py
Continuous execution loop for the trading bot.

Key changes vs original:
  - Respects NYSE market hours (computed in ET, DST-safe)
  - Removes hardcoded --dry-run (honours the flag passed by the user)
  - Uses sys.executable instead of a fragile relative venv path
  - Sleeps until market open rather than burning CPU at 3 AM
  - Stops automatically at market close
"""

import sys
import time
import subprocess
import argparse
from logger import log
from market_hours import is_market_open, minutes_until_open, market_status_str


def run_trading_loop(
    symbol: str = "AAPL",
    interval_seconds: int = 3600,
    dry_run: bool = False,
    iterations: int | None = None,
) -> None:
    mode = "DRY-RUN" if dry_run else "LIVE"
    log(f"Trading loop starting -- symbol={symbol}, interval={interval_seconds}s, mode={mode}")

    count = 0

    try:
        while True:
            # -- Market-hours gate ------------------------------------------------
            if not is_market_open():
                wait_mins = minutes_until_open()
                log(f"Market closed. {market_status_str()} | Next open in ~{wait_mins} min.")
                # Sleep in 60-second chunks so Ctrl+C still works
                for _ in range(min(wait_mins, 60)):
                    time.sleep(60)
                continue   # Re-check; don't proceed to trade

            # -- Build the subprocess command -------------------------------------
            cmd = [sys.executable, "main.py", "--symbol", symbol]
            if dry_run:
                cmd.append("--dry-run")

            log(f"--- Iteration {count + 1} | {market_status_str()} ---")

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.stdout:
                    print(result.stdout, end="")
                if result.returncode != 0:
                    log(f"main.py exited with code {result.returncode}")
                if result.stderr:
                    log(f"STDERR from main.py:\n{result.stderr.strip()}")
            except subprocess.TimeoutExpired:
                log("main.py timed out after 120 seconds -- skipping this iteration.")
            except Exception as exc:
                log(f"Failed to launch main.py: {exc}")

            count += 1
            if iterations and count >= iterations:
                log(f"Reached {iterations} iterations. Stopping.")
                break

            log(f"Sleeping {interval_seconds}s until next cycle...")
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        log("Loop stopped by user (Ctrl+C).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trading bot in a continuous loop.")
    parser.add_argument("--symbol",     type=str,  default="AAPL",  help="Stock symbol to trade.")
    parser.add_argument("--interval",   type=int,  default=3600,    help="Seconds between iterations.")
    parser.add_argument("--dry-run",    action="store_true",         help="Pass --dry-run to main.py (no orders placed).")
    parser.add_argument("--iterations", type=int,  default=None,    help="Stop after N iterations (default: run forever).")
    args = parser.parse_args()

    run_trading_loop(args.symbol, args.interval, args.dry_run, args.iterations)
