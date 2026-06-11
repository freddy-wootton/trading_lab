"""
Handles fetching historical stock data from Alpaca.
"""
import datetime

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from config import API_KEY, SECRET_KEY
from logger import log

_client = None


def _get_client():
    """Create the Alpaca data client only when the first API call needs it."""
    global _client
    if _client is None:
        _client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    return _client


def get_daily_bars(symbol, days=30):
    """
    Fetch daily bar data (Open, High, Low, Close, Volume) for a given symbol.
    Returns a data frame of the last 'days' bars.
    """
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        limit=days,
        feed="iex"  # Explicitly use IEX for Free Tier users
    )

    # Fetch results and return them as a pandas dataframe
    bars = _get_client().get_stock_bars(request_params)
    return bars.df


def get_intraday_bars(symbol, lookback_bars=100):
    """
    Fetch 5-minute intraday bar data for a given symbol.
    Returns a data frame of the last 'lookback_bars' candles.
    """
    log(f"API call: fetching {lookback_bars} bars for {symbol}")
    end_date = datetime.datetime.now() - datetime.timedelta(minutes=16)
    calendar_days = max(3, int(lookback_bars / 78) + 3)
    start_date = end_date - datetime.timedelta(days=calendar_days)
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start_date,
        end=end_date,
        limit=lookback_bars,
        feed="iex"  # Explicitly use IEX for Free Tier users
    )

    bars = _get_client().get_stock_bars(request_params)
    return bars.df.tail(lookback_bars)
