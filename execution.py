"""
execution.py
Submits market orders to Alpaca Paper Trading.

Changes:
  - TradingClient created once as a module-level singleton (not per call)
  - TimeInForce changed from GTC to DAY (correct for equity market orders)
  - Guards against qty <= 0
"""

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import API_KEY, SECRET_KEY
from logger import log

# Module-level singleton -- created once when this module is imported
_client = TradingClient(API_KEY, SECRET_KEY, paper=True)


def submit_order(symbol: str, side: str, qty: int = 1):
    """
    Submit a market order to Alpaca.

    Args:
        symbol: Ticker symbol, e.g. 'AAPL'
        side:   'buy' or 'sell'
        qty:    Number of whole shares (must be >= 1)

    Returns:
        The Alpaca order object, or None on failure.
    """
    if qty <= 0:
        log(f"submit_order: qty={qty} -- skipping order for {symbol}")
        return None

    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    log(f"Submitting {order_side.value} order: {qty} x {symbol}")

    try:
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,   # DAY is correct for equity market orders
        )
        order = _client.submit_order(order_data=market_order_data)
        log(f"Order accepted. ID: {order.id}")
        return order
    except Exception as exc:
        log(f"Order failed for {symbol}: {exc}")
        return None