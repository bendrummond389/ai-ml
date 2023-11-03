import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout


def build_rnn_model(
    input_shape, units=50, dropout_rate=0.2, output_units=4
):
    """
    Build a simple RNN model for time series prediction.
    """
    model = Sequential()

    # RNN layer
    model.add(SimpleRNN(units=units, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout_rate))

    # Dense output layer - No changes here, but ensure output_units matches what you're predicting
    model.add(Dense(units=output_units))
    
    optimizer = tf.keras.optimizers.legacy.Adam()

    # Compile the model
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    return model


