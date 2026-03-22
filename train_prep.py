"""
Data preparation script to fetch historical data for training the ML model.
"""
import pandas as pd
import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from config import API_KEY, SECRET_KEY
from logger import log

# Initialize the Data Client
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Broad market symbols to build a diverse dataset (Top 30 Nasdaq 100)
NASDAQ_SUBSYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "PEP", "COST",
    "ADBE", "CSCO", "NFLX", "TMUS", "AMD", "QCOM", "INTC", "AMGN", "HON", "TXN",
    "AMAT", "BKNG", "ISRG", "ADI", "MU", "PANW", "VRTX", "REGN", "PYPL", "FISV"
]

def fetch_training_data(symbols=None, days=365):
    """
    Fetches historical daily bars for multiple symbols and saves them to 'training_data.csv'.
    Uses a 15-minute delay and pauses between requests to respect API rate limits.
    """
    if symbols is None:
        symbols = NASDAQ_SUBSYMBOLS
        
    log(f"Fetching {days} days of training data for {len(symbols)} symbols...")
    
    all_data = []
    
    # Alpaca Free Tier has a 15-minute delay for market data
    end_date = datetime.datetime.now() - datetime.timedelta(minutes=16)
    start_date = end_date - datetime.timedelta(days=days)
    
    for symbol in symbols:
        try:
            log(f"-> Downloading {symbol}...")
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
                feed=DataFeed.IEX # Free tier uses IEX feed
            )
            
            # Fetch and convert to DataFrame
            bars = client.get_stock_bars(request_params)
            df = bars.df
            
            if not df.empty:
                # Reset index to make 'timestamp' a column
                df.reset_index(inplace=True)
                all_data.append(df)
            
            # Small delay to avoid hammering the API
            import time
            time.sleep(0.5)
            
        except Exception as e:
            log(f"Error fetching {symbol}: {e}")
            continue
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Save to CSV for the training loop in ml_strategy.py
        final_df.to_csv("training_data.csv", index=False)
        log(f"Successfully saved combined training data to training_data.csv ({len(final_df)} total rows)")
    else:
        log("No data fetched. CSV not updated.")

if __name__ == "__main__":
    # Fetch data for the full list defined above
    fetch_training_data()

