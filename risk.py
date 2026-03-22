"""
Risk management logic for calculating safe trade sizes.
"""
def calculate_position_size(account_balance, price, risk_percent=0.02):
    """
    Calculates the number of shares to buy based on a risk percentage of the total balance.
    Default risk is 2% of total cash per trade.
    """
    # Total dollar amount to risk
    position_value = account_balance * risk_percent
    
    # Number of shares based on current price
    qty = int(position_value / price)
    
    # Ensure at least 1 share is traded if within risk limits (or handle zero)
    return max(qty, 1)