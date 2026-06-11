"""
training_scheduler.py
Schedules daily model retraining at 02:00 ET (not local time).
"""

import time
from datetime import datetime
import zoneinfo

from train_prep import fetch_training_data
from ml_strategy import train_model
from logger import log

ET = zoneinfo.ZoneInfo("America/New_York")


def retrain():
    log("Scheduled retraining starting...")
    try:
        fetch_training_data(days=365)
        train_model()
        log("Retraining complete.")
    except Exception as exc:
        log(f"Retraining failed: {exc}")


def run_scheduler():
    log("Training scheduler started. Will retrain daily at 02:00 ET.")
    # schedule library uses local time -- use a manual loop with
    # timezone-aware check for reliability across DST transitions
    while True:
        now_et = datetime.now(tz=ET)
        if now_et.hour == 2 and now_et.minute == 0:
            retrain()
            time.sleep(61)   # Prevent double-firing within the same minute
        time.sleep(30)


if __name__ == "__main__":
    run_scheduler()
