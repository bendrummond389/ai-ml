import os
import tensorflow as tf


from preprocessing.data_processor import process_data
from preprocessing.sequence_creator import create_sequences
from models.rnn_model import build_rnn_model
from training.train import train_model
from data_collection.recent_data_collector import download_recent_stock_data
from utils.config import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import warnings


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.ticker import MaxNLocator





def visualize_performance(
    model_path, history_path, test_sequences, test_labels, scaler
):
    """
    Visualizes the model's performance by plotting the actual vs. predicted values.
    Only the 'Close' price is plotted for actual and predicted values.
    """

    print(f"loading model from {model_path}")
    # Load the model
    model = tf.keras.models.load_model(model_path)
    print("model loaded")

    # Load the training history
    history = np.load(history_path, allow_pickle=True).item()

    # Predict the test set
    predicted_labels = model.predict(test_sequences)

    # Assume that the 'Close' price is the first column in the labels
    # Adjust the indices if necessary
    close_index = 0

    # Invert the scaling for a better interpretation
    predicted_close = scaler.inverse_transform(predicted_labels)[:, close_index]
    actual_close = scaler.inverse_transform(test_labels)[:, close_index]

    # Define a style
    plt.style.use("ggplot")

    # Create a figure with a constrained layout
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Plot the training and validation loss
    axs[0].plot(history["loss"], label="Training Loss", linewidth=2)
    axs[0].plot(history["val_loss"], label="Validation Loss", linewidth=2)
    axs[0].set_title("Training and Validation Loss Over Epochs", fontsize=14)
    axs[0].set_xlabel("Epochs", fontsize=12)
    axs[0].set_ylabel("Loss", fontsize=12)
    axs[0].legend(frameon=True)
    axs[0].grid(True)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot the predicted vs actual values for 'Close' price
    axs[1].plot(actual_close, label="Actual Close Price", linewidth=2)
    axs[1].plot(predicted_close, label="Predicted Close Price", alpha=0.7, linewidth=2)
    axs[1].set_title("Predicted vs Actual Close Price", fontsize=14)
    axs[1].set_xlabel("Sample Index", fontsize=12)
    axs[1].set_ylabel("Price", fontsize=12)
    axs[1].legend(frameon=True)
    axs[1].grid(True)

    # Optionally, set x-ticks to be less dense
    axs[1].xaxis.set_major_locator(
        MaxNLocator(nbins=10)
    )  # Adjust the number of bins as needed

    # Rotate x-tick labels if they're overlapping
    plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right")

    # Show the plot
    plt.show()


def train():
    # download_recent_stock_data("AAPL")

    # Define file paths
    historical_data_path = "data/historical/AAPL_2010-01-01_to_2023-01-01.csv"
    recent_data_path = "data/recent/AAPL_recent.csv"

    # Process historical data
    (
        historical_features_train,
        historical_labels_train,
        historical_features_test,
        historical_labels_test,
        _,
        _,
    ) = process_data(historical_data_path)

    # Process recent data
    (
        recent_features_train,
        recent_labels_train,
        recent_features_test,
        recent_labels_test,
        _,
        _,
    ) = process_data(recent_data_path)
    

    # Create sequences from historical data
    historical_sequences, historical_sequence_labels = create_sequences(
        historical_features_train, historical_labels_train, SEQUENCE_LENGTH
    )

    # Create sequences from recent data
    recent_sequences, recent_sequence_labels = create_sequences(
        recent_features_train, recent_labels_train, SEQUENCE_LENGTH
    )

    # Permute the data
    historical_permutation = np.random.permutation(historical_sequences.shape[0])
    historical_sequences = historical_sequences[historical_permutation]
    historical_sequence_labels = historical_sequence_labels[historical_permutation]  # This line corrected

    # Build the RNN model
    input_shape = (
        SEQUENCE_LENGTH,
        historical_features_train.shape[1],
    )  # assuming features_train is 2D: (samples, features)
    model = build_rnn_model(input_shape=input_shape)

    # Train the RNN model with historical data and validate on recent data
    history = train_model(
        train_data=(historical_sequences, historical_sequence_labels),
        validation_data=(recent_sequences, recent_sequence_labels),
        model=model,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    # Save model and history if needed
    model_path = os.path.join(MODEL_SAVE_PATH, "rnn_model.h5")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*You are saving your model as an HDF5 file.*",
    )
    model.save(model_path)

    np.save(os.path.join(MODEL_SAVE_PATH, "training_history.npy"), history.history)


# Now, in your main function, after the training is complete, call the visualize_performance function:
def main():
    # print(tf.__version__)
    # train()

    recent_data_path = "data/recent/AAPL_recent.csv"
    recent_train_data, recent_test_data, recent_scaler = process_data(recent_data_path)

    # Define file paths
    model_path = os.path.join(MODEL_SAVE_PATH, "rnn_model.h5")
    history_path = os.path.join(MODEL_SAVE_PATH, "training_history.npy")
    recent_sequences, recent_labels = create_sequences(
        recent_train_data, SEQUENCE_LENGTH
    )

    visualize_performance(
        model_path, history_path, recent_sequences, recent_labels, recent_scaler
    )


if __name__ == "__main__":
    main()
