import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np


def plot_validation_data(file_path, validation_sequences, validation_labels, scaler):
    model = keras.models.load_model(file_path)

    predicted = model.predict(validation_sequences)

    predicted_prices = scaler.inverse_transform(
        np.concatenate(
            (
                np.zeros((predicted.shape[0], validation_sequences.shape[2] - 1)),
                predicted,
            ),
            axis=1,
        )
    )[:, -1]

    actual_prices = scaler.inverse_transform(
        np.concatenate(
            (
                np.zeros(
                    (validation_labels.shape[0], validation_sequences.shape[2] - 1)
                ),
                validation_labels.reshape(-1, 1),
            ),
            axis=1,
        )
    )[:, -1]

    # Get the sorted indices based on actual prices
    sorted_indices = np.argsort(actual_prices)
    sorted_actual_prices = actual_prices[sorted_indices]
    sorted_predicted_prices = predicted_prices[sorted_indices]

    plt.figure(figsize=(14, 7))
    plt.plot(sorted_actual_prices, color="black", label="Actual Prices")
    plt.plot(
        sorted_predicted_prices, color="green", linestyle="--", label="Predicted Prices"
    )
    plt.title("Comparison of Sorted Actual and Predicted Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
