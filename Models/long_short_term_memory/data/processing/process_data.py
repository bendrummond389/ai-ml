from typing import Tuple, Any
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ta.trend import EMAIndicator


# load data from a csv and return a pandas dataframe
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, index_col="Date", parse_dates=True)


# fill missing values in the dataframe with various possible methods
def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    data.interpolate()
    # data.fillna(method='ffill', inplace=True)
    # data.fillna(method='bfill', inplace=True)
    return data


# choose what features from our data set we would like to use
def select_features(data: pd.DataFrame) -> pd.DataFrame:
    # Selecting 'Open', 'High', 'Low', 'Close', 'Volume' for now
    return data[["Open", "High", "Low", "Close", "Volume"]]


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
    return data


# scale data (normalize) using MinMaxScaler
def scale_data(data: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def create_time_steps(
    data: np.ndarray, time_step: int
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []

    # Assuming 'Close' price is always the fourth column
    close_index = 3  # Zero-based index, so 3 means fourth column

    for i in range(time_step, len(data)):
        # Append the past 'time_step' days of features to X
        X.append(data[i - time_step : i, :])
        # Append the current day's 'Close' price to y
        y.append(data[i, close_index])

    return np.array(X), np.array(y)


# a way to visualize the indicators
def plot_data(data: pd.DataFrame):
    plt.figure(figsize=(14, 7))
    plt.plot(data["Close"].tail(50), label="Close Price", color="blue")
    plt.plot(data["EMA_20"].tail(50), label="EMA 20", color="red", linestyle="--")

    plt.title("Stock Price with EMA Indicator")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    plt.show()


def process_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, Any]:
    data = load_data(file_path)
    data = fill_missing_values(data)
    features = select_features(data)
    with_indicators = add_technical_indicators(features)
    scaled_data, scaler = scale_data(with_indicators)
    sequences, output = create_time_steps(scaled_data, 20)

    return sequences, output, scaler
