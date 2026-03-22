"""
Portfolio management: Interaction with Alpaca account and positions.
"""
from alpaca.trading.client import TradingClient
from config import API_KEY, SECRET_KEY

# Initialize the Alpaca Trading Client (defaults to Paper if configured in config.py)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

def get_account_balance():
    """
    Fetch the current buying power/cash balance of the Alpaca account.
    """
    account = trading_client.get_account()
    return float(account.cash)

def get_position(symbol):
    """
    Get the quantity of an open position for a specific symbol.
    Returns 0 if no position exists.
    """
    try:
        position = trading_client.get_open_position(symbol)
        return int(position.qty)
    except:
        # If no position is found, return 0
        return 0