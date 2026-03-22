"""
Handles fetching historical stock data from Alpaca.
"""
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import API_KEY, SECRET_KEY

# Initialize the Alpaca Data Client
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

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
    bars = client.get_stock_bars(request_params)
    return bars.df