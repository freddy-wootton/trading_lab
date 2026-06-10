# ML Trading Bot Lab

> A machine-learning–driven algorithmic trading bot built with **PyTorch (LSTM)**, the **Alpaca Paper Trading API**, **SQLAlchemy/SQLite**, and a **Streamlit dashboard**.

---

## Project Structure

### Core Trading Pipeline
| File | Purpose |
|---|---|
| `main.py` | Orchestrates data fetch → signal generation → risk sizing → order execution |
| `ml_strategy.py` | LSTM neural net, feature engineering (RSI, SMA, lags), MinMaxScaler, training & inference |
| `execution.py` | Order submission to the Alpaca Paper Trading API |
| `risk.py` | Position-sizing logic (2% of balance per trade; returns 0 when balance is too small) |
| `portfolio.py` | Account balance and position queries |
| `data.py` | Fetches live daily bars via Alpaca |
| `database.py` | SQLAlchemy models (`trade_results`, `predictions`, `model_checkpoints`) + helpers |
| `config.py` | Loads API keys from `.env` |
| `logger.py` | Centralized logging to console and `trading.log` |

### ML Training
| File | Purpose |
|---|---|
| `train_prep.py` | Downloads 1 year of OHLCV data for 30 Nasdaq symbols |
| `ml_strategy.py` | `train_model()` — trains the LSTM and saves `model_lstm.pth` + `scaler.pkl` |
| `training_scheduler.py` | Runs `train_model()` automatically every day at 02:00 using `schedule` |

### Automation & UI
| File | Purpose |
|---|---|
| `run_loop.py` | Runs `main.py` at a configurable interval (default 1 h) |
| `dashboard.py` | Streamlit dashboard: metrics, equity curve, P&L chart, predictions vs actuals |
| `backtester.py` | `compute_metrics(df)` — total P&L, win rate, Sharpe ratio, max drawdown, equity curve |
| `start.bat` | Launches the bot **and** dashboard in two separate terminal windows |
| `setup_env.bat` | Creates `.venv` and installs all dependencies from `requirements.txt` |

---

## Getting Started

### 1. Set up the environment
```bat
setup_env.bat
```

### 2. Configure API keys

> [!IMPORTANT]
> **Never commit your `.env` file.** It is listed in `.gitignore`.

Create a `.env` file in the project root based on this template:
```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```
Keys are obtained from [https://app.alpaca.markets](https://app.alpaca.markets) (Paper Trading section).

### 3. Download training data
```bat
.venv\Scripts\python train_prep.py
```
Downloads ~1 year of daily bars for 30 Nasdaq 100 symbols and saves `training_data.csv`.

### 4. Train the model
```bat
.venv\Scripts\python ml_strategy.py
```
Trains the LSTM and saves `model_lstm.pth` and `scaler.pkl` to the project root.

### 5. Run the bot + dashboard (recommended)
```bat
start.bat
```
This opens **two terminal windows**: one for the trading bot and one for the Streamlit dashboard.  
Or launch individually:
```bat
# Trading bot (one iteration)
.venv\Scripts\python main.py --symbol AAPL

# Continuous loop (default 1-hour interval, dry-run mode)
.venv\Scripts\python run_loop.py --symbol AAPL

# Dashboard only
.venv\Scripts\streamlit run dashboard.py

# Daily retraining scheduler
.venv\Scripts\python training_scheduler.py
```

---

## Dashboard

The Streamlit dashboard (`dashboard.py`) is available at **http://localhost:8501** after launch.

| Section | Description |
|---|---|
| **Metric Cards** | Total P&L, Win Rate, Sharpe Ratio, Max Drawdown |
| **Equity Curve** | Cumulative P&L over all trades |
| **P&L per Trade** | Bar chart coloured by profit/loss |
| **Predictions vs Actual** | ML model's forecasted price overlaid on actual close |
| **Recent Trades Table** | Last 20 trades with full metadata |
| **Sidebar** | Filter by symbol and date range |

The dashboard **auto-refreshes every 60 seconds** to show the latest data.

---

## Key Design Notes

- **No credentials in source control.** API keys live exclusively in `.env` (gitignored).
- **Feature normalisation.** The LSTM is trained on MinMaxScaler-normalised features. The fitted scaler is saved to `scaler.pkl` and applied at inference time.
- **Small balance protection.** If `2% × balance < share price`, `calculate_position_size()` returns `0` and the trade is skipped with a log warning.
- **Paper trading only.** `PAPER = True` is hardcoded in `config.py`.
