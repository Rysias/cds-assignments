"""
Functionality for loading image data for the train and test sets
"""
import numpy as np
from typing import Tuple


def preprocess_image_data(images: np.ndarray) -> np.ndarray:
    """
    Preprocesses the images in a dataset.

    Args:
        images (np.ndarray): The images to preprocess.

    Returns:
        np.ndarray: The preprocessed images.
    """
    images = images.astype("float32")
    images /= 255
    return images


def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the CIFAR10 dataset from Keras.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            The first element is the training images, the second element is the
            training labels, the third element is the test images, and the fourth
            element is the test labels.
    """
    from tensorflow.keras.datasets import cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return x_train, y_train, x_test, y_test
