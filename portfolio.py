from alpaca.trading.client import TradingClient
from config import API_KEY, SECRET_KEY, PAPER

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)

def get_account_balance():
    account = trading_client.get_account()
    return float(account.cash)

def get_position(symbol):
    try:
        position = trading_client.get_open_position(symbol)
        return int(position.qty)
    except:
        return 0