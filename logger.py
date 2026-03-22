"""
Custom logging utility to print and store trading activities.
"""
def log(message):
    """
    Prints a message to the console and appends it to 'trading.log' with a timestamp.
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    with open("trading.log", "a") as f:
        f.write(formatted_message + "\n")