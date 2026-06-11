"""
Continuous execution loop for the trading bot.
Runs the main trading logic at a set interval (e.g., every hour).
"""
import time
import subprocess
import argparse
from logger import log

def run_trading_loop(symbol="AAPL", interval_seconds=10, iterations=None):
    """
    Infinite loop that executes 'main.py' at regular intervals.
    """
    count = 0
    log(f"Starting continuous trading loop for {symbol} (Interval: {interval_seconds}s)")
    
    try:
        while True:
            log(f"--- Iteration {count + 1} Starting ---")
            
            # Execute main.py as a subprocess to keep the loop isolated from potential crashes
            # We use --dry-run by default for safety in the loop
            cmd = [".venv/Scripts/python", "main.py", "--symbol", symbol, "--dry-run"]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    log(f"Error in main.py: {result.stderr}")
            except Exception as e:
                log(f"Failed to run main.py: {e}")
            
            count += 1
            if iterations and count >= iterations:
                log("Reached requested iteration limit. Stopping.")
                break
                
            log(f"Iteration {count} complete. Sleeping for {interval_seconds} seconds...")
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        log("Loop stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trading bot in a loop.")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol to trade.")
    parser.add_argument("--interval", type=int, default=3600, help="Interval between runs in seconds.")
    parser.add_argument("--iterations", type=int, default=None, help="Number of iterations to run (None for infinite).")
    
    args = parser.parse_args()
    run_trading_loop(args.symbol, args.interval, args.iterations)
