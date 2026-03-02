from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from config import API_KEY, SECRET_KEY

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def get_daily_bars(symbol):
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=60)
    )

    bars = data_client.get_stock_bars(request)
    return bars.df