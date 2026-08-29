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

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # the function must return a tuple of two lists: images and labels
    images = []
    labels = []
    # iterate through each category directory and load images and labels
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        # check if the category directory exists
        if not os.path.isdir(category_dir):
            continue
        # iterate through each image file in the category directory
        for filename in os.listdir(category_dir):
            filepath = os.path.join(category_dir, filename)
            image = cv2.imread(filepath)
            # check if the image was loaded successfully
            if image is None:
                continue
            # resize the image to the desired dimensions
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            # append the image and label to the respective lists
            images.append(image)
            labels.append(category)
     # and the whole point of this function
     # returning the images and labels as a tuple
    return images, labels

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        # The first convolutional layer. 
        # It slides 32 different tiny 3x3 filters across the image, 
        # each one learning to detect a simple pattern (an edge, a curve, a blob of color)
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        # Shrink the image by taking the strongest signal out of every 2x2 block of pixels. 
        # This reduces the amount of data to process and 
        # makes the network less sensitive to a sign being slightly shifted or rotated.
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # A second, deeper convolutional layer with more filters (64). 
        # Since it's working on the simplified output of the first layer, 
        # it can start combining simple patterns into more complex ones.
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        # Shrinks the data again, same idea as before
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # squashes it into one long 1D list of numbers, because the next layers need a flat input.
        tf.keras.layers.Flatten(),
        # A fully connected layer with 128 neurons.
        # Each neuron looks at the entire image and decides whether it sees a particular pattern.
        tf.keras.layers.Dense(128, activation="relu"),
        # During training, this randomly switches off 50% of the neurons each round. 
        # It sounds destructive, but it actually prevents the network from memorizing 
        # the training images too rigidly, forcing it to learn more general patterns instead.
        tf.keras.layers.Dropout(0.5),
        # The output layer. It has one neuron for each category,
        # and uses the softmax activation function to turn the raw numbers into probabilities.
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
