"""
risk.py
Position sizing: risks a fixed percentage of account balance per trade.
"""

from logger import log


def calculate_position_size(
    account_balance: float,
    price: float,
    risk_percent: float = 0.02,
) -> int:
    """
    Return the number of whole shares to buy, capped at the risk budget.

    Example:
        account_balance = 10_000, price = 291.48, risk_percent = 0.02
        risk_budget = 200.00
        qty = int(200 / 291.48) = 0  -> skip trade (correct behaviour)

    The previous code did max(qty, 1) which would force a $291 trade
    on a $200 risk budget -- a 29% position instead of 2%.
    """
    if price <= 0:
        log("calculate_position_size: price must be > 0")
        return 0

    risk_budget = account_balance * risk_percent
    qty = int(risk_budget / price)

    if qty < 1:
        log(
            f"Position size is 0 -- balance=${account_balance:.2f}, "
            f"price=${price:.2f}, risk_budget=${risk_budget:.2f}. "
            f"Trade skipped."
        )
        return 0

    return qty