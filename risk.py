def calculate_position_size(account_balance, price, risk_percent=0.02):
    position_value = account_balance * risk_percent
    qty = int(position_value / price)
    return max(qty, 1)