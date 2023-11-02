import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.datasets import mnist
from models.cnn_model import create_simple_cnn
from keras import callbacks
from visualize import plot_feature_maps

def train_and_evaluate():
    # Load the dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the pixel values of the images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Reshape the data to fit the model
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    # One-hot encode the labels
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # Create the CNN model
    model = create_simple_cnn(input_shape=(28, 28, 1), num_classes=10)

    # Use the legacy version of the Adam optimizer as suggested by TensorFlow
    optimizer = tf.keras.optimizers.legacy.Adam()

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Display the model's architecture
    model.summary()

    # Define early stopping
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    # Train the model with early stopping
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[early_stopping])
    
    test_image_to_visualize = test_images[0:1] # This creates a batch with one image
    plot_feature_maps(model, test_image_to_visualize)

# Add the code below to your main.py to call the train_and_evaluate function
if __name__ == "__main__":
    train_and_evaluate()
