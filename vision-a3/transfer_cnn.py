"""
- Load the CIFAR10 dataset
- Use VGG16 to perform feature extraction
- Train a classifier 
- Save plots of the loss and accuracy 
- Save the classification report
"""
import src.load_data as load_data
import src.vgg16 as vgg16


INPUT_SIZE = (32, 32, 3)
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def main():
    # Load the CIFAR10 dataset (NB: Make sure input size is right!)
    x_train, y_train, x_test, y_test = load_data.load_cifar10()

    # Create the model
    model = vgg16.finetuneable_vgg16(INPUT_SIZE, LEARNING_RATE)

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
    )


if __name__ == "__main__":
    main()
