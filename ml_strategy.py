import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from logger import log

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
    import os
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        log(f"Model loaded from {path}")
        return True
    return False

def train_model(data_path="training_data.csv", epochs=100, lr=0.001, seq_length=10):
    """
    Trains the LSTM model using historical data.
    Organizes data into sequences of length 'seq_length' to predict the next price.
    """
    df = pd.read_csv(data_path)
    if df.empty:
        log("Training data is empty!")
        return
    
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
    data = df[all_features].values.astype(np.float32)
    labels = df['close'].values.astype(np.float32)
    
    # Create sequences: (Samples, Seq_Length, Features)
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(labels[i + seq_length])
    
    X = torch.tensor(np.array(X)).float()
    y = torch.tensor(np.array(y)).float().view(-1, 1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMPricePredictor(input_size=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    log(f"Starting LSTM training on {device}...")
    model.train()
    X, y = X.to(device), y.to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            log(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    save_model(model)
    return model

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
        log(f"Visualization saved to latest_prediction.png")
    except Exception as e:
        log(f"Failed to generate plot: {e}")

def predict_signal(df: pd.DataFrame, seq_length=10) -> tuple[str, float]:
    """
    Generates a trading signal for the latest data point using the trained LSTM.
    Prepares a sequence of the last 'seq_length' points for inference.
    """
    if len(df) < seq_length + 30: # Need enough for sequences and SMA30
        return "none", 0.0

    # Engineer features for the whole dataframe to ensure latest point has indicators
    df = df.copy()
    df['rsi'] = compute_rsi(df['close'])
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    for j in range(1, 7):
        df[f'lag_{j}'] = df['close'].shift(j)
    
    df.dropna(inplace=True)
    
    feature_cols = ['close', 'rsi', 'sma_10', 'sma_30'] + [f'lag_{j}' for j in range(1, 7)]
    
    # Get latest sequence
    latest_seq = df[feature_cols].tail(seq_length).values.astype(np.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_tensor = torch.tensor([latest_seq]).float().to(device)
    
    model = LSTMPricePredictor(input_size=10).to(device)
    if not load_model(model):
        log("No pre-trained LSTM model found. Using random weights.")
    
    model.eval()
    with torch.no_grad():
        prediction_scalar = model(features_tensor).item()
    
    current_price = df['close'].iloc[-1]
    
    # Signal logic based on price prediction threshold
    signal = "flat"
    if prediction_scalar > current_price * 1.005: 
        signal = "long"
    elif prediction_scalar < current_price * 0.995:
        signal = "short"
    
    # Generate the prediction chart (optional visualization)
    plot_forecast(df, float(prediction_scalar))
    
    # Return signal and prediction
    return signal, float(prediction_scalar)


if __name__ == "__main__":
    # Internal test for training
    train_model()

