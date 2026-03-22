"""
Execution layer: Submits trade orders to the Alpaca API.
"""
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import API_KEY, SECRET_KEY
from logger import log

def submit_order(symbol: str, side: str, qty: int = 1):
    """
    Submits a market order to Alpaca Paper Trading.
    'side' should be 'buy' or 'sell'.
    Handles order construction and error reporting.
    """
    # Initialize the Trading Client (Paper trading is default for these keys/account)
    client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    
    # Map the requested side to Alpaca's OrderSide enum
    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    
    log(f"Submitting {order_side} order for {qty} share(s) of {symbol}...")
    
    try:
        # Prepare the market order request
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.GTC # Good 'Til Cancelled
        )
        
        # Dispatch the order to Alpaca's servers
        market_order = client.submit_order(order_data=market_order_data)
        log(f"Order submitted successfully. Order ID: {market_order.id}")
        return market_order
    except Exception as e:
        # Log any errors (e.g., market closed, insufficient funds)
        log(f"Failed to submit order for {symbol}: {e}")
        return None