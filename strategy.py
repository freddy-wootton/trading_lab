def moving_average_crossover(df):
    closes = df['close']

    if len(closes) < 30:
        return None

    sma_10 = closes.tail(10).mean()
    sma_30 = closes.tail(30).mean()

    if sma_10 > sma_30:
        return "long"
    else:
        return "flat"