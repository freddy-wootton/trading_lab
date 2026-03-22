# Project Status Summary: Trading Lab v1.0

## Completed Milestones
- **Infrastructure**: Full PyTorch integration with **LSTM** architecture and **SQLAlchemy** (SQLite) trade logging.
- **Data Pipeline**: Automated fetch for **30 Nasdaq tickers** (7,500 rows) with `train_prep.py`.
- **Strategy**: 10-feature model using Price Lags, **RSI**, and **SMAs**.
- **Execution**: Paper-ready submission via `execution.py` with dynamic **2% risk-based position sizing**.
- **Automation**: Continuous trading loop via `run_loop.py` and live **Price vs. Prediction plotting** in `ml_strategy.py`.
- **Packaging**: One-click EXE build script (`build_snapshot.py`) for easy laptop-to-desktop transfer.

## Active Status
- **Model**: Trained on 7,500 samples; weights saved in `model_lstm.pth`.
- **Database**: `trading_lab.db` initialized and ready for logging.
- **Visuals**: `latest_prediction.png` generated on every run.

## Next Phase (Future)
1. **Model Scaling**: Transition to a **1-billion parameter Transformer** model using the `scaling_roadmap.md`.
2. **Feature Expansion**: Integrate sentiment analysis and cross-asset correlations.
3. **GUI Development**: Wrap the EXE in a dashboard to show real-time performance and PnL.

- **Active Logic**: LSTM (Batch=1, Seq=10, Feat=10); Alpaca IEX; 2% Risk/Trade.
