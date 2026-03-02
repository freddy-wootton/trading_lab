"""
trading_lab - Starter Script
Author: Freddy
Description: Basic project structure for a trading bot / strategy lab
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# ==========================
# LOAD ENVIRONMENT VARIABLES
# ==========================
load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

# ==========================
# CONFIG
# ==========================
SYMBOL = "AAPL"
FAST_MA = 10
SLOW_MA = 30
CHECK_INTERVAL = 60  # seconds


# ==========================
# CLIENT SETUP
# ==========================
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ==========================
# FUNCTIONS
# ==========================

def get_data(symbol):
    end = datetime.utcnow()
    start = end - timedelta(days=5)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end
    )

    bars = data_client.get_stock_bars(request).df

    if bars.empty:
        return None

    df = bars.reset_index()
    return df


def calculate_mas(df):
    df["fast_ma"] = df["close"].rolling(FAST_MA).mean()
    df["slow_ma"] = df["close"].rolling(SLOW_MA).mean()
    return df


def holding_position(symbol):
    positions = trading_client.get_all_positions()
    for p in positions:
        if p.symbol == symbol:
            return True
    return False


def place_order(symbol, qty, side):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY
    )
    trading_client.submit_order(order)
    logging.info(f"{side} order submitted for {qty} shares of {symbol}")


# ==========================
# MAIN LOOP
# ==========================

logging.info("Bot started.")

while True:
    try:
        clock = trading_client.get_clock()

        if not clock.is_open:
            logging.info("Market closed. Sleeping 5 minutes.")
            time.sleep(300)
            continue

        logging.info("Fetching data...")
        df = get_data(SYMBOL)

        if df is None or len(df) < SLOW_MA:
            logging.warning("Not enough data yet.")
            time.sleep(CHECK_INTERVAL)
            continue

        df = calculate_mas(df)

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        logging.info(
            f"Price: {latest['close']:.2f} | "
            f"Fast MA: {latest['fast_ma']:.2f} | "
            f"Slow MA: {latest['slow_ma']:.2f}"
        )

        # BUY SIGNAL
        if previous["fast_ma"] < previous["slow_ma"] and latest["fast_ma"] > latest["slow_ma"]:
            if not holding_position(SYMBOL):
                logging.info("BUY signal detected.")
                place_order(SYMBOL, 5, OrderSide.BUY)
            else:
                logging.info("Already holding. No buy.")

        # SELL SIGNAL
        elif previous["fast_ma"] > previous["slow_ma"] and latest["fast_ma"] < latest["slow_ma"]:
            if holding_position(SYMBOL):
                logging.info("SELL signal detected.")
                place_order(SYMBOL, 5, OrderSide.SELL)
            else:
                logging.info("No position to sell.")

        else:
            logging.info("No signal.")

        logging.info("Sleeping...\n")
        time.sleep(CHECK_INTERVAL)

    except Exception as e:
        logging.error(f"Error: {e}")
        time.sleep(60)