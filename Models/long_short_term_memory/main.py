from data.collection.collect_data import download_aapl_data
from data.processing.process_data import process_data, get_last_sequence
from models.lstm_model_v1 import create_lstm_model, create_advanced_lstm_model
from training.training import train_lstm_model
from visualize.visualize_performance import plot_validation_data
from tensorflow import keras
import numpy as np


if __name__ == "__main__":
    download_aapl_data()

    # PREDICT_NEXT_PRICE

    # last_sequence, scaler = get_last_sequence(file_path="data/storage/AAPL_training_data.csv")

    # model = keras.saving.load_model("models/storage/lstm_model.h5")

    # predicted_scaled = model.predict(last_sequence)

    # # Create a dummy array with the same shape as your input features
    # dummy_array = np.zeros((predicted_scaled.shape[0], 15))

    # # Fill the dummy array with your predicted 'Close' price at the correct column index
    # close_index = 3  # This should match the index used in 'create_time_steps' function
    # dummy_array[:, close_index] = predicted_scaled.ravel()  # Flatten the array if necessary

    # # Now inverse transform the dummy array
    # predicted_price = scaler.inverse_transform(dummy_array)[:, close_index]  # Extract the 'Close' price column

    # print(predicted_price)

    X_train, X_val, y_train, y_val, scaler = process_data(
        file_path="data/storage/AAPL_training_data.csv"
    )

    input_shape = (X_train.shape[1], X_train.shape[2])

    model = create_advanced_lstm_model(input_shape)
    model, history = train_lstm_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    plot_validation_data(
        "models/storage/lstm_model.h5",
        X_val,
        y_val,
        scaler,
    )
