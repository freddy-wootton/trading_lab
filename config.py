"""
config.py
Centralised configuration. All settings and constants live here.
Raises on startup if required environment variables are missing.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def _require(key: str) -> str:
    """Load an env var and raise clearly if it is absent or empty."""
    value = os.getenv(key, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Create a .env file in the project root with {key}=your_value"
        )
    return value

# Alpaca credentials (loaded lazily so dry-run tests don't need keys)
def get_api_key() -> str:
    return _require("ALPACA_API_KEY")

def get_secret_key() -> str:
    return _require("ALPACA_SECRET_KEY")

# Keep these for backward compatibility with files that do 'from config import API_KEY'
API_KEY    = os.getenv("ALPACA_API_KEY", "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# Trading settings
PAPER          = True        # Always paper-trade; set to False only with great care
DEFAULT_SYMBOL = "AAPL"
FAST_MA        = 10
SLOW_MA        = 30
RISK_PERCENT   = 0.02        # 2 % of account balance risked per trade
SIGNAL_THRESHOLD = 0.005     # 0.5 % price move needed to trigger long/short
TRADE_INTERVAL_SECONDS = 3600  # How often run_loop.py fires (1 hour)