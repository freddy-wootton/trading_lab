"""
ML strategy module: LSTM model definition, training, feature engineering, and prediction.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from config import SIGNAL_THRESHOLD
from logger import log

SCALER_PATH = "scaler.pkl"


class LSTMPricePredictor(nn.Module):
    """
    Long Short-Term Memory (LSTM) network for time-series price prediction.
    LSTMs are better than simple linear models for capturing temporal dependencies.
    """
    def __init__(self, input_size=10, hidden_size=64, num_layers=2):
        super(LSTMPricePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer: input_size is features per timestep
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def compute_rsi(series, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a price series.
    RSI is a momentum oscillator that measures the speed and change of price movements.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def save_model(model, path="model_lstm.pth"):
    """Save model weights to a file."""
    torch.save(model.state_dict(), path)
    log(f"Model saved to {path}")


def load_model(model, path="model_lstm.pth"):
    """Load model weights from a file if it exists."""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        log(f"Model loaded from {path}")
        return True
    return False


def train_model(data_path="training_data.csv", epochs=100, lr=0.001, seq_length=10):
    """
    Trains the LSTM model using historical data.
    Organizes data into sequences of length 'seq_length' to predict the next price.
    Fits a MinMaxScaler on training features and saves it to scaler.pkl.

    Returns:
        Trained model and (train_loss, val_loss) from the final epoch as a tuple,
        or None if training data is unavailable.
    """
    df = pd.read_csv(data_path)
    if df.empty:
        log("Training data is empty!")
        return None

    # Feature engineering: Add RSI and Moving Averages
    df['rsi'] = compute_rsi(df['close'])
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df.dropna(inplace=True)

    # Prepare features (10 total)
    feature_cols = ['close', 'rsi', 'sma_10', 'sma_30']

    # Add 6 lagged closing prices as additional features
    for j in range(1, 7):
        df[f'lag_{j}'] = df['close'].shift(j)
    df.dropna(inplace=True)

    all_features = feature_cols + [f'lag_{j}' for j in range(1, 7)]
    raw_data = df[all_features].values.astype(np.float32)
    labels = df['close'].values.astype(np.float32)

    # Fit a MinMaxScaler on raw feature data and save it for use in predict_signal
    scaler = MinMaxScaler()
    data = scaler.fit_transform(raw_data)
    joblib.dump(scaler, SCALER_PATH)
    log(f"Feature scaler fitted and saved to {SCALER_PATH}")

    # Create sequences: (Samples, Seq_Length, Features)
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i: i + seq_length])
        y.append(labels[i + seq_length])

    X = torch.tensor(np.array(X)).float()
    y = torch.tensor(np.array(y)).float().view(-1, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMPricePredictor(input_size=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    log(f"Starting LSTM training on {device}...")

    # 80/20 train/validation split
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size].to(device), X[train_size:].to(device)
    y_train, y_val = y[:train_size].to(device), y[train_size:].to(device)

    final_train_loss = 0.0
    final_val_loss = 0.0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), y_val).item()
            log(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
            final_train_loss = loss.item()
            final_val_loss = val_loss

    save_model(model)
    return model, (final_train_loss, final_val_loss)


def plot_forecast(df, prediction, symbol="Stock"):
    """
    Generates a simple price chart showing historical trend and the ML prediction.
    Saves the plot to 'latest_prediction.png'.
    """
    try:
        plt.figure(figsize=(10, 6))

        # Plot last 30 days of actual prices
        lookback = min(len(df), 30)
        subset = df.tail(lookback).copy()

        plt.plot(subset.index, subset['close'], label='Historical Price', color='blue', marker='o')

        # Plot prediction (as a single point after the last historical point)
        pred_index = subset.index[-1] + 1
        plt.scatter(pred_index, prediction, color='red', label='Model Prediction', zorder=5)
        plt.axhline(y=prediction, color='red', linestyle='--', alpha=0.3)

        plt.title(f"{symbol} Price Forecast (ML Snapshot)")
        plt.xlabel("Days (Discrete)")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig("latest_prediction.png")
        plt.close()
        log("Visualization saved to latest_prediction.png")
    except Exception as e:
        log(f"Failed to generate plot: {e}")


def predict_signal(df: pd.DataFrame, seq_length=10, model=None) -> tuple[str, float]:
    """
    Generates a trading signal for the latest data point using the trained LSTM.
    Prepares a sequence of the last 'seq_length' points for inference.

    Args:
        df: DataFrame of price history with at least a 'close' column.
        seq_length: Number of time steps in each input sequence.
        model: Optional pre-loaded LSTMPricePredictor. If None, loads from disk.
               Pass a pre-loaded model to avoid repeated disk I/O when calling
               this function in a tight loop.
    """
    if len(df) < seq_length + 30:  # Need enough rows for sequences + SMA30
        return "none", 0.0

    # Engineer features for the whole dataframe so the latest row has valid indicators
    df = df.copy()
    df['rsi'] = compute_rsi(df['close'])
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    for j in range(1, 7):
        df[f'lag_{j}'] = df['close'].shift(j)

    df.dropna(inplace=True)

    feature_cols = ['close', 'rsi', 'sma_10', 'sma_30'] + [f'lag_{j}' for j in range(1, 7)]

    # Get the latest sequence of raw feature values
    latest_seq = df[feature_cols].tail(seq_length).values.astype(np.float32)

    # Apply the MinMaxScaler that was fitted during training (if it exists)
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            latest_seq = scaler.transform(latest_seq).astype(np.float32)
        except Exception as e:
            log(f"Warning: Failed to apply scaler from {SCALER_PATH}: {e}. Proceeding without scaling.")
    else:
        log(f"Warning: Scaler not found at {SCALER_PATH}. Proceeding without scaling. Run train_model() to generate it.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_tensor = torch.tensor(np.array([latest_seq])).float().to(device)

    # Use the provided model or fall back to loading from disk
    if model is None:
        model = LSTMPricePredictor(input_size=10).to(device)
        if not load_model(model):
            log("No pre-trained LSTM model found. Using random weights.")
    else:
        model = model.to(device)

    model.eval()
    with torch.no_grad():
        prediction_scalar = model(features_tensor).item()

    current_price = df['close'].iloc[-1]

    # Signal logic based on the configured price prediction threshold.
    signal = "flat"
    if prediction_scalar > current_price * (1 + SIGNAL_THRESHOLD):
        signal = "long"
    elif prediction_scalar < current_price * (1 - SIGNAL_THRESHOLD):
        signal = "short"

    # Generate the prediction chart (optional visualization)
    plot_forecast(df, float(prediction_scalar))

    return signal, float(prediction_scalar)


if __name__ == "__main__":
    # Internal test for training
    train_model(data_path="training_data_intraday.csv", epochs=50)
