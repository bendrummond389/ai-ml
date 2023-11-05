from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils.config import BATCH_SIZE, EPOCHS


def train_lstm_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        "models/storage/lstm_model.h5", monitor="val_loss", save_best_only=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, min_lr=0.00001, verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1,
    )

    return model, history
