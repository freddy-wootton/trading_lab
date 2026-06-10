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
    
    # If position value is less than the share price, qty will be 0.
    # Return 0 to signal that the trade should be skipped — do NOT force 1 share.
    return qty