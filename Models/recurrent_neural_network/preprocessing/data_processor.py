import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator
import numpy as np


def load_data(file_path):
    """
    Load data from CSV file.
    """
    data = pd.read_csv(file_path)
    return data


def clean_data(data):
    """
    Clean data by handling missing values.
    """
    missing_values = data.isnull().sum()
    print("Missing values in each column:\n", missing_values)

    if missing_values.any():
        # Forward-fill missing values
        data.fillna(method="ffill", inplace=True)

    missing_values_after = data.isnull().sum()
    print("Missing values after handling in each column:\n", missing_values_after)

    return data


def select_features(data, include_volume=True):
    """
    Select and optionally transform the features from the data.
    """
    features = data[["Close"]].copy()  # Start with just the 'Close' prices

    # High and Low prices
    features["High"] = data["High"]
    features["Low"] = data["Low"]

    # Include volume data if specified
    if include_volume:
        features["Volume"] = data["Volume"]

    # Technical Indicators
    # Moving Averages
    features["SMA_20"] = data["Close"].rolling(window=20).mean()
    features["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

    # Momentum Indicators
    features["RSI"] = RSIIndicator(data["Close"]).rsi()

    # MACD
    macd = MACD(data["Close"])
    features["MACD"] = macd.macd()
    features["MACD_signal"] = macd.macd_signal()

    # Volatility Indicators
    bollinger = BollingerBands(data["Close"])
    features["Bollinger_mavg"] = bollinger.bollinger_mavg()
    features["Bollinger_hband"] = bollinger.bollinger_hband()
    features["Bollinger_lband"] = bollinger.bollinger_lband()

    # Volume Indicators
    if include_volume:
        features["OBV"] = OnBalanceVolumeIndicator(
            data["Close"], data["Volume"]
        ).on_balance_volume()

    return features.dropna()


def scale_data(data, scaler_type="minmax"):
    """
    Scale data using MinMaxScaler or StandardScaler.
    """
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def split_data(data, test_size=0.2):
    """
    Split data into training and test sets.
    """
    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data


def create_labels(data, include_volume=True):
    """
    Create labels from the data.
    """
    labels = data[["High", "Low", "Close"]].copy()  # Primary labels for prediction

    if include_volume:
        labels["Volume"] = data["Volume"]  # Include volume if specified

    return labels


def process_data(file_path, include_volume=True, scaler_type="minmax", test_size=0.2):
    """
    Process data by loading, cleaning, selecting features, scaling, splitting, and creating labels.
    """
    data = load_data(file_path)
    data = clean_data(data)
    features = select_features(data, include_volume=include_volume)
    labels = create_labels(data, include_volume=include_volume)

    # Scale features
    scaled_features, features_scaler = scale_data(features, scaler_type=scaler_type)
    # Scale labels separately
    scaled_labels, labels_scaler = scale_data(labels, scaler_type=scaler_type)

    # Split features and labels into training and test sets
    features_train, features_test = split_data(scaled_features, test_size=test_size)
    labels_train, labels_test = split_data(scaled_labels, test_size=test_size)

    return (
        features_train,
        labels_train,
        features_test,
        labels_test,
        features_scaler,
        labels_scaler,
    )


if __name__ == "__main__":
    file_path = "path/to/your/csv/file.csv"
    (
        features_train,
        labels_train,
        features_test,
        labels_test,
        features_scaler,
        labels_scaler,
    ) = process_data(file_path)
