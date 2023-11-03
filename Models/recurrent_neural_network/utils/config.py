# Data configuration
SEQUENCE_LENGTH = 60
FEATURES = [
    "Close",
    "High",
    "Low",
    "Volume",
]
# Model configuration
BATCH_SIZE = 32
EPOCHS = 50
LSTM_UNITS = 50
DENSE_UNITS = 1
DROPOUT_RATE = 0.2

# Training configuration
MODEL_SAVE_PATH = "models/cache"
HISTORY_SAVE_PATH = "history/cache"

# Paths to data
HISTORICAL_DATA_PATH = "data/historical/AAPL_2010-01-01_to_2023-01-01.csv"
RECENT_DATA_PATH = "data/recent/AAPL_recent.csv"
