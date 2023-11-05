from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Conv1D
from keras.regularizers import l1_l2
from keras.optimizers.legacy import Adam


def create_lstm_model(input_shape, units=300, dropout_rate=0.1, reg_lambda=0.01):
    model = Sequential()

    # First LSTM layer with L1/L2 regularization and return sequences true for stacking
    model.add(
        Bidirectional(
            LSTM(
                units=units,
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l1_l2(l1=reg_lambda, l2=reg_lambda),
            )
        )
    )
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    # Second LSTM layer with regularization
    model.add(
        Bidirectional(
            LSTM(
                units=units,
                return_sequences=False,
                kernel_regularizer=l1_l2(l1=reg_lambda, l2=reg_lambda),
            )
        )
    )
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    # Output layer with one neuron for regression prediction
    model.add(Dense(units=1))

    # Compiling the model with the Adam optimizer and learning rate scheduling
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mean_absolute_error")

    return model


def create_advanced_lstm_model(input_shape):
    model = Sequential()

    # Adding a convolution YOLO
    model.add(
        Bidirectional(LSTM(units=512, return_sequences=True), input_shape=input_shape)
    )
    model.add(Dropout(0.1))

    # Adding another LSTM layer with more units
    model.add(LSTM(units=128, return_sequences=False))

    model.add(Dense(units=128))
    model.add(Dense(units=128))

    # Output layer remains the same
    model.add(Dense(units=1))

    model.summary()

    # Compiling the model with a potentially smaller learning rate
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    return model
