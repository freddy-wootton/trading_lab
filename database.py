"""
Database layer for logging trade results using SQLAlchemy and SQLite.
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Database Configuration
DATABASE_URL = "sqlite:///trading_lab.db"

# Define the base class for SQLAlchemy models
Base = declarative_base()

class TradeResult(Base):
    """
    Database model representing a single trade event and its prediction data.
    """
    __tablename__ = "trade_results"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    symbol = Column(String)
    signal = Column(String) # 'long', 'short', or 'flat'
    close_price = Column(Float)
    fast_ma = Column(Float)
    slow_ma = Column(Float)
    prediction = Column(Float) # The model's predicted price

    def __init__(self, symbol, signal, close_price, fast_ma, slow_ma, prediction):
        """Initialize the record with trade data."""
        self.symbol = symbol
        self.signal = signal
        self.close_price = close_price
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.prediction = prediction


# Setup the SQLite engine and session factory
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create all tables defined in the Base class if they don't exist."""
    Base.metadata.create_all(bind=engine)

def log_trade(symbol: str, signal: str, close_price: float, fast_ma: float = None, slow_ma: float = None, prediction: float = None):
    """
    Save a new trade record to the database.
    """
    db = SessionLocal()
    try:
        new_result = TradeResult(
            symbol=symbol,
            signal=signal,
            close_price=close_price,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            prediction=prediction
        )
        db.add(new_result)
        db.commit()
        db.refresh(new_result)
        return new_result
    finally:
        db.close()
