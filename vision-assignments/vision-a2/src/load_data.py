# import mnist data from keras
from typing import Tuple
import numpy as np
import logging

# basic logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


def load_image_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pass


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset from Keras.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            The first element is the training images, the second element is the
            training labels, the third element is the test images, and the fourth
            element is the test labels.
    """
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    x_train = flatten_image_data(x_train)
    x_test = flatten_image_data(x_test)
    return x_train, y_train, x_test, y_test


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
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    x_train = flatten_image_data(x_train)
    x_test = flatten_image_data(x_test)
    return x_train, y_train, x_test, y_test


def flatten_image_data(images: np.ndarray) -> np.ndarray:
    """
    Flattens the images in a dataset.

    Args:
        images (np.ndarray): The images to flatten.

    Returns:
        np.ndarray: The flattened images.
    """
    return images.reshape(images.shape[0], -1)


def load_dataset(dataset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the dataset from Keras.

    Args:
        dataset (str): The dataset to load.

    Returns:
        Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            A function that returns the dataset.
    """
    all_dataloaders = {"mnist": load_mnist, "cifar10": load_cifar10}
    try:
        logging.info("Loading data...")
        return all_dataloaders[dataset]()
    except KeyError:
        raise ValueError(
            f"You can only choose from {all_dataloaders.keys()}: {dataset} chosen"
        )

