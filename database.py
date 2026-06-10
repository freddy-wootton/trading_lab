"""
Database layer: SQLAlchemy models and helper functions for logging trades,
predictions, and model checkpoints to a local SQLite database.
"""
from __future__ import annotations

import csv
import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# Engine / session setup
# ---------------------------------------------------------------------------
DATABASE_URL = "sqlite:///trading_lab.db"

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TradeResult(Base):
    """
    Represents a single trade event.
    The legacy table name (trade_results) is retained for backward compatibility.
    """
    __tablename__ = "trade_results"

    id            = Column(Integer, primary_key=True)
    timestamp     = Column(DateTime, default=datetime.datetime.utcnow)
    symbol        = Column(String)
    side          = Column(String)          # 'buy' / 'sell' / None
    signal        = Column(String)          # 'long', 'short', or 'flat'
    entry_price   = Column(Float)           # close price at signal time
    exit_price    = Column(Float)           # nullable — filled when position closed
    qty           = Column(Integer)
    pnl           = Column(Float)           # nullable — filled on exit
    ml_prediction = Column(Float)          # model's predicted next price
    fast_ma       = Column(Float)
    slow_ma       = Column(Float)
    rsi           = Column(Float)

    # ------------------------------------------------------------------
    # Keep the old positional __init__ signature so existing call sites
    # (log_trade) continue to work unchanged.
    # ------------------------------------------------------------------
    def __init__(
        self,
        symbol: str,
        signal: str,
        close_price: float,
        fast_ma: float | None = None,
        slow_ma: float | None = None,
        prediction: float | None = None,
        side: str | None = None,
        qty: int | None = None,
        exit_price: float | None = None,
        pnl: float | None = None,
        rsi: float | None = None,
    ):
        self.symbol        = symbol
        self.signal        = signal
        self.entry_price   = close_price
        self.fast_ma       = fast_ma
        self.slow_ma       = slow_ma
        self.ml_prediction = prediction
        self.side          = side
        self.qty           = qty
        self.exit_price    = exit_price
        self.pnl           = pnl
        self.rsi           = rsi


class Prediction(Base):
    """Records each ML model prediction for later analysis."""
    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True)
    timestamp       = Column(DateTime, default=datetime.datetime.utcnow)
    symbol          = Column(String)
    close_price     = Column(Float)
    predicted_price = Column(Float)
    signal          = Column(String)


class ModelCheckpoint(Base):
    """Stores metadata about model training runs."""
    __tablename__ = "model_checkpoints"

    id         = Column(Integer, primary_key=True)
    timestamp  = Column(DateTime, default=datetime.datetime.utcnow)
    epoch      = Column(Integer)
    train_loss = Column(Float)
    val_loss   = Column(Float)


# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables defined in Base if they don't already exist."""
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def log_trade(
    symbol: str,
    signal: str,
    close_price: float,
    fast_ma: float | None = None,
    slow_ma: float | None = None,
    prediction: float | None = None,
    side: str | None = None,
    qty: int | None = None,
    exit_price: float | None = None,
    pnl: float | None = None,
    rsi: float | None = None,
) -> TradeResult:
    """Save a new trade record to the database and return it."""
    db = SessionLocal()
    try:
        record = TradeResult(
            symbol=symbol,
            signal=signal,
            close_price=close_price,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            prediction=prediction,
            side=side,
            qty=qty,
            exit_price=exit_price,
            pnl=pnl,
            rsi=rsi,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()


def log_prediction(
    symbol: str,
    close_price: float,
    predicted_price: float,
    signal: str,
) -> Prediction:
    """Save an ML prediction record."""
    db = SessionLocal()
    try:
        record = Prediction(
            symbol=symbol,
            close_price=close_price,
            predicted_price=predicted_price,
            signal=signal,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()


def log_model_checkpoint(epoch: int, train_loss: float, val_loss: float) -> ModelCheckpoint:
    """Save a model training checkpoint record."""
    db = SessionLocal()
    try:
        record = ModelCheckpoint(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_all_trades() -> pd.DataFrame:
    """Return all trade records as a pandas DataFrame."""
    db = SessionLocal()
    try:
        rows = db.query(TradeResult).order_by(TradeResult.timestamp).all()
        if not rows:
            return pd.DataFrame()
        data = [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "symbol": r.symbol,
                "side": r.side,
                "signal": r.signal,
                "entry_price": r.entry_price,
                "exit_price": r.exit_price,
                "qty": r.qty,
                "pnl": r.pnl,
                "ml_prediction": r.ml_prediction,
                "fast_ma": r.fast_ma,
                "slow_ma": r.slow_ma,
                "rsi": r.rsi,
            }
            for r in rows
        ]
        return pd.DataFrame(data)
    finally:
        db.close()


def get_all_predictions() -> pd.DataFrame:
    """Return all prediction records as a pandas DataFrame."""
    db = SessionLocal()
    try:
        rows = db.query(Prediction).order_by(Prediction.timestamp).all()
        if not rows:
            return pd.DataFrame()
        data = [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "symbol": r.symbol,
                "close_price": r.close_price,
                "predicted_price": r.predicted_price,
                "signal": r.signal,
            }
            for r in rows
        ]
        return pd.DataFrame(data)
    finally:
        db.close()


def export_trades_to_csv(filename: str = "trades_export.csv") -> Path:
    """Export all trade records to a CSV file and return its path."""
    df = get_all_trades()
    path = Path(filename)
    df.to_csv(path, index=False)
    return path
