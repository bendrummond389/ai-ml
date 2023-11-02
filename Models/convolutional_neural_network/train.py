# train.py

import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.datasets import mnist
from models.cnn_model import create_simple_cnn

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

  # Confirm if everything is loaded correctly
  print("Training set shape:", train_images.shape)
  print("Test set shape:", test_images.shape)
  
  # Create the CNN model
  model = create_simple_cnn(input_shape=(28, 28, 1), num_classes=10)

  # Compile the model
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  # Display the model's architecture
  model.summary()
  
  # Train the model
  model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Add the code below to your main.py to call the train_and_evaluate function
if __name__ == "__main__":
    train_and_evaluate()
