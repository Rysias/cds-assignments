"""
- Load the CIFAR10 dataset
- Use VGG16 to perform feature extraction
- Train a classifier 
- Save plots of the loss and accuracy 
- Save the classification report
"""
import src.load_data as load_data
import tensorflow as tf


def main():
    # Load the CIFAR10 dataset (NB: Make sure input size is right!)
    x_train, y_train, x_test, y_test = load_data.load_cifar10()

    # Load the VGG16 model
    base_model = tf.keras.applications.vgg16.VGG16(
        include_top=False, weights="imagenet", classifier_activation="softmax",
    )

    base_model.trainable = False


if __name__ == "__main__":
    main()
