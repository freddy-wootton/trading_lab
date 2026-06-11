"""
logger.py
Centralised logging to console and trading.log.
All timestamps are in UTC so logs are unambiguous across DST transitions.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path(__file__).parent / "trading.log"

# Configure the standard library logger once
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",          # We format ourselves for the file
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
_logger = logging.getLogger("trading_lab")


def log(message: str) -> None:
    """Log a message with a UTC timestamp to both console and trading.log."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    _logger.info(f"[{ts}] {message}")