# ML Trading Bot Lab

A functional machine learning-based trading algorithm using **PyTorch**, **Alpaca API**, and **SQLAlchemy**.

## Project Structure

This project is organized into modular Python files, each handling a specific part of the trading pipeline:

### Core Files
- **[main.py](file:///c:/Users/fredd/OneDrive%20-%20University%20of%20Exeter/trading_lab/main.py)**: The central entry point. Orchestrates data fetching, signal generation, and execution.
- **[ml_strategy.py](file:///c:/Users/fredd/OneDrive%20-%20University%20of%20Exeter/trading_lab/ml_strategy.py)**: Contains the **LSTM Neural Network** (PyTorch) logic, feature engineering (RSI, SMA), and price prediction.
- **[execution.py](file:///c:/Users/fredd/OneDrive%20-%20University%20of%20Exeter/trading_lab/execution.py)**: Handles order submission to the Alpaca Paper Trading API.
- **[database.py](file:///c:/Users/fredd/OneDrive%20-%20University%20of%20Exeter/trading_lab/database.py)**: Logs every trade, prediction, and technical indicator to a local SQLite database (`trading_lab.db`).

### Utilities
- **[data.py](file:///c:/Users/fredd/OneDrive%20-%20University%20of%20Exeter/trading_lab/data.py)**: Fetches historical daily bars for real-time analysis.
- **[train_prep.py](file:///c:/Users/fredd/OneDrive%20-%20University%20of%20Exeter/trading_lab/train_prep.py)**: Downloads 1 year of historical data for model training.
- **[portfolio.py](file:///c:/Users/fredd/OneDrive%20-%20University%20of%20Exeter/trading_lab/portfolio.py)**: Manages account balance and position queries.
- **[risk.py](file:///c:/Users/fredd/OneDrive%20-%20University%20of%20Exeter/trading_lab/risk.py)**: Implements position sizing logic (risking 2% of balance per trade).
- **[config.py](file:///c:/Users/fredd/OneDrive%20-%20University%20of%20Exeter/trading_lab/config.py)**: Manages API keys and global settings.
- **[logger.py](file:///c:/Users/fredd/OneDrive%20-%20University%20of%20Exeter/trading_lab/logger.py)**: Centralized logging to both the console and `trading.log`.

### Automated Execution
- **[run_loop.py](file:///c:/Users/fredd/OneDrive%20-%20University%20of%20Exeter/trading_lab/run_loop.py)**: Use this to run the bot continuously (e.g., every hour) during market hours.

## Getting Started
1. Run `setup_env.bat` to install dependencies.
2. Update `.env` with your Alpaca API Keys.
3. Run `python train_prep.py` to get data.
4. Run `python ml_strategy.py` to train the model.
5. Run `python main.py` or `python run_loop.py` to start trading!
