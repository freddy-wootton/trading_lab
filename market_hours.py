"""
market_hours.py
Timezone-aware US market hours checks for a UK-based operator.

The NYSE opens 09:30-16:00 US Eastern Time (ET).
From the UK:
  - During US Standard Time (EST, UTC-5): UK sees this as 14:30-21:00 GMT
  - During US Daylight Time (EDT, UTC-4): UK sees this as 13:30-20:00 BST
  
Gap-week problem:
  US clocks change on the 2nd Sunday of March and 1st Sunday of November.
  UK clocks change on the last Sunday of March and last Sunday of October.
  During the 2-3 week gap after US changes but before UK changes (in spring),
  and the 1-week gap after UK changes but before US changes (in autumn),
  the offset between UK and ET is different from the usual +/-1 hour.

Solution: Always work in UTC internally. Convert to ET using zoneinfo.
"""

from datetime import datetime, time, timedelta
import zoneinfo

# These are the canonical timezone objects -- use them everywhere.
ET = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")
UK = zoneinfo.ZoneInfo("Europe/London")

# NYSE trading session (ET)
MARKET_OPEN_ET  = time(9, 30)
MARKET_CLOSE_ET = time(16, 0)


def now_et() -> datetime:
    """Return the current moment as a timezone-aware datetime in US/Eastern."""
    return datetime.now(tz=UTC).astimezone(ET)


def now_utc() -> datetime:
    """Return the current moment as a timezone-aware UTC datetime."""
    return datetime.now(tz=UTC)


def now_uk() -> datetime:
    """Return the current moment as a timezone-aware datetime in Europe/London."""
    return datetime.now(tz=UTC).astimezone(UK)


def is_market_open() -> bool:
    """
    Return True only if the NYSE is currently open for trading.
    Checks:
      1. It is a weekday (Mon-Fri)
      2. The ET clock time is between 09:30 and 16:00
    Does NOT account for NYSE holidays (add a holiday calendar library
    such as 'trading-calendars' or 'pandas-market-calendars' for that).
    """
    et_now = now_et()
    if et_now.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    current_time = et_now.time()
    return MARKET_OPEN_ET <= current_time < MARKET_CLOSE_ET


def minutes_until_open() -> int:
    """
    Return the number of minutes until the next NYSE open.
    Returns 0 if the market is currently open.
    Useful for run_loop.py to sleep until the market opens.
    """
    if is_market_open():
        return 0

    et_now = now_et()
    # Build today's open datetime in ET
    today_open = et_now.replace(
        hour=MARKET_OPEN_ET.hour,
        minute=MARKET_OPEN_ET.minute,
        second=0,
        microsecond=0
    )

    if et_now < today_open and et_now.weekday() < 5:
        # Market hasn't opened yet today
        delta = today_open - et_now
    else:
        # Market is closed; find next weekday open
        days_ahead = 1
        while True:
            candidate = et_now + timedelta(days=days_ahead)
            candidate = candidate.replace(
                hour=MARKET_OPEN_ET.hour,
                minute=MARKET_OPEN_ET.minute,
                second=0,
                microsecond=0,
            )
            # Re-attach ET timezone after replace
            candidate = candidate.astimezone(ET)
            if candidate.weekday() < 5:
                break
            days_ahead += 1
        delta = candidate - et_now

    return max(0, int(delta.total_seconds() / 60))


def market_status_str() -> str:
    """Human-readable market status for logging (shows both ET and UK times)."""
    et_now = now_et()
    uk_now = now_uk()
    status = "OPEN" if is_market_open() else "CLOSED"
    return (
        f"Market {status} | ET: {et_now.strftime('%Y-%m-%d %H:%M %Z')} | "
        f"UK: {uk_now.strftime('%Y-%m-%d %H:%M %Z')}"
    )
