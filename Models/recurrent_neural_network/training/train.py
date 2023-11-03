import os
import numpy as np
import tensorflow as tf
from models.rnn_model import build_rnn_model
from preprocessing.data_processor import process_data
from utils.config import BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH
from tensorflow.keras.callbacks import EarlyStopping


def train_model(train_data, validation_data, model, batch_size, epochs):
    train_sequences, train_labels = train_data
    validation_sequences, validation_labels = validation_data
    
    print(f"Train sequences shape: {train_data[0].shape}")
    print(f"Train labels shape: {train_data[1].shape}")
    print(f"Validation sequences shape: {validation_data[0].shape}")
    print(f"Validation labels shape: {validation_data[1].shape}")
    
    early_stopping_callback = EarlyStopping(
    monitor='val_mae',  # Monitor validation MAE
    patience=2,         # Number of epochs with no improvement after which training will be stopped
    mode='min',         # The direction is "minimize" since we want to minimize MAE
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored metric
    )

    
    history = model.fit(
        train_sequences,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(validation_sequences, validation_labels),
        callbacks=[early_stopping_callback]
    )
    return history
