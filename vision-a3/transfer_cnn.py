"""
- Load the CIFAR10 dataset
- Use VGG16 to perform feature extraction
- Train a classifier 
- Save plots of the loss and accuracy 
- Save the classification report
"""
import src.load_data as load_data


def main():
    # Load the CIFAR10 dataset
    x_train, y_train, x_test, y_test = load_data.load_cifar10()
