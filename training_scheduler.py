"""
Daily training scheduler for the Trading Lab.

Runs an infinite loop that:
  - Every day at 02:00, re-downloads the latest 1 year of training data
    and retrains the LSTM model.
  - Saves a checkpoint record to the database on success.
  - Logs failures with full exception details so the scheduler never crashes.

Start with:
    python training_scheduler.py
"""
from __future__ import annotations

import time
import traceback
from datetime import datetime

import schedule

from database import init_db, log_model_checkpoint
from logger import log
from ml_strategy import train_model
from train_prep import fetch_training_data


def retrain_job() -> None:
    """
    Full retrain pipeline: download fresh data → train → save checkpoint.
    Exceptions are caught and logged so the scheduler loop stays alive.
    """
    log("=== Scheduled retraining started ===")
    try:
        # Step 1: Download the latest 1 year of historical data
        log("Downloading latest training data (1 year)...")
        fetch_training_data(days=365)
        log("Training data download complete.")

        # Step 2: Retrain the model
        log("Starting LSTM model training...")
        result = train_model(epochs=100)

        if result is None:
            log("ERROR: train_model() returned None — training data may be empty.")
            return

        trained_model, (train_loss, val_loss) = result

        # Step 3: Persist a checkpoint record
        log_model_checkpoint(epoch=100, train_loss=train_loss, val_loss=val_loss)
        log(
            f"Retraining complete. "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

    except Exception:
        log("ERROR: Retraining job failed with exception:")
        log(traceback.format_exc())

    log("=== Scheduled retraining finished ===")


def main() -> None:
    """Set up the schedule and run the loop."""
    init_db()
    log("Training scheduler starting. Retraining scheduled daily at 02:00.")

    schedule.every().day.at("02:00").do(retrain_job)

    # Also run once immediately on startup so you can verify connectivity
    log("Running initial retrain on startup...")
    retrain_job()

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
