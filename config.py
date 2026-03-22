"""
Configuration management: Loads API keys from environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca API Credentials
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Trading Settings
PAPER = True  # Always use paper trading for safety
DEFAULT_SYMBOL = "AAPL"
FAST_MA = 10
SLOW_MA = 30