import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Verify the command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Load image data and their labels
    images, labels = load_data(sys.argv[1])
    print(images[228].shape)  # Debug: Print shape of one image

    # Split the data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Build and compile the neural network
    model = get_model()

    # Train the model using the training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate the model's performance with testing data
    model.evaluate(x_test, y_test, verbose=2)

    # Save the trained model if a filename is provided
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load images and their labels from `data_dir`.

    The directory should contain subdirectories named 0 to NUM_CATEGORIES - 1.
    Each subdirectory contains images for a specific category.

    Returns:
        A tuple `(images, labels)`:
        - `images` is a list of resized images as numpy arrays.
        - `labels` is a list of integers, each representing the category.
    """
    dirs = os.listdir(data_dir)
    images = []
    labels = []

    for dirname in dirs:
        imgs = os.listdir(os.path.join(data_dir, dirname))
        for img in imgs:
            # Read the image and resize it to the required dimensions
            image = cv2.imread(os.path.join(data_dir, dirname, img))
            resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            images.append(resized)
            labels.append(int(dirname))  # Use the folder name as the label
    return (images, labels)


def get_model():
    """
    Create and return a compiled convolutional neural network (CNN).

    The CNN takes input images of shape `(IMG_WIDTH, IMG_HEIGHT, 3)` and
    outputs a prediction for one of the NUM_CATEGORIES classes.

    Layers include:
        - Convolutional and pooling layers for feature extraction.
        - A fully connected (dense) layer with dropout for regularization.
        - An output layer with NUM_CATEGORIES units and softmax activation.
    """
    model = tf.keras.models.Sequential([

        # First convolutional layer with 128 filters and a 5x5 kernel
        tf.keras.layers.Conv2D(
            128, (5, 5), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Max pooling layer with 2x2 pool size to reduce spatial dimensions
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten layer to convert 2D feature maps into 1D vectors
        tf.keras.layers.Flatten(),

        # Dense layer with dropout to reduce overfitting
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Output layer with softmax activation for category prediction
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
