from typing import Tuple, Any
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils.config import TIME_STEPS

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange


# load data from a csv and return a pandas dataframe
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, index_col="Date", parse_dates=True)


def add_day_of_week(data: pd.DataFrame) -> pd.DataFrame:
    # Add a 'Day_of_Week' column
    data["Day_of_Week"] = data.index.dayofweek

    # One-hot encode the 'Day_of_Week' column
    day_of_week_one_hot = pd.get_dummies(data["Day_of_Week"], prefix="DOW")

    # Drop the original 'Day_of_Week' column
    data = data.drop("Day_of_Week", axis=1)

    # Concatenate the one-hot encoded columns to the original dataframe
    data = pd.concat([data, day_of_week_one_hot], axis=1)

    return data


# fill missing values in the dataframe with various possible methods
def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    data.interpolate()
    # data.fillna(method='ffill', inplace=True)
    # data.fillna(method='bfill', inplace=True)
    return data


# choose what features from our data set we would like to use
def select_features(data: pd.DataFrame) -> pd.DataFrame:
    # Selecting 'Open', 'High', 'Low', 'Close', 'Volume' for now
    return data[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "DOW_0",
            "DOW_1",
            "DOW_2",
            "DOW_3",
            "DOW_4",
        ]
    ]


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    # EMA
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

    # RSI
    data["RSI"] = RSIIndicator(close=data["Close"]).rsi()

    # MACD (keeping it as it is)
    macd = MACD(close=data["Close"])
    data["MACD"] = macd.macd()

    # On-Balance Volume (OBV)
    obv = OnBalanceVolumeIndicator(close=data["Close"], volume=data["Volume"])
    data["OBV"] = obv.on_balance_volume()

    # Average True Range (ATR)
    atr = AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"])
    data["ATR"] = atr.average_true_range()

    # Drop NaN values in case the indicators introduce them
    data.dropna(inplace=True)

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


def get_last_sequence(file_path) -> Tuple[np.ndarray, Any]:
    data = load_data(file_path).tail(200)
    data = fill_missing_values(data)
    data = add_day_of_week(data)
    features = select_features(data)
    with_indicators = add_technical_indicators(features)
    print(with_indicators.tail(10))
    scaled_data, scaler = scale_data(with_indicators)

    if len(scaled_data) >= TIME_STEPS:
        last_sequence = scaled_data[-TIME_STEPS:]
    else:
        # If not, this is a problem. We can't generate a sequence.
        raise ValueError("Not enough data to generate the last sequence.")

    # Reshape the last sequence to match the input shape for the LSTM model,
    # which should be (1, time_steps, number_of_features)
    last_sequence = last_sequence.reshape((1, TIME_STEPS, last_sequence.shape[1]))

    return last_sequence, scaler


def process_data(file_path: str, validation_size=0.2):
    data = load_data(file_path)
    data = fill_missing_values(data)
    data = add_day_of_week(data)
    features = select_features(data)
    with_indicators = add_technical_indicators(features)
    scaled_data, scaler = scale_data(with_indicators)
    X, y = create_time_steps(scaled_data, TIME_STEPS)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_size, shuffle=True
    )

    return X_train, X_val, y_train, y_val, scaler
