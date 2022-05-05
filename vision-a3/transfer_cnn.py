"""
- Load the CIFAR10 dataset
- Use VGG16 to perform feature extraction
- Train a classifier 
- Save plots of the loss and accuracy 
- Save the classification report
"""
import argparse
import src.load_data as load_data
import src.vgg16 as vgg16
import src.report as report
import src.evaluate as evaluate
from pathlib import Path

# Input size is fixed to 32x32 in CIFAR10
INPUT_SIZE = (32, 32, 3)
OUTPUT_DIR = Path("output")


def main(args: argparse.Namespace) -> None:
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs

    # Load the CIFAR10 dataset (NB: Make sure input size is right!)
    x_train, y_train, x_test, y_test = load_data.load_cifar10()

    # Create the model
    model = vgg16.finetuneable_vgg16(INPUT_SIZE, LEARNING_RATE)

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
    )

    # Create classification report
    classification_results = evaluate.evaluate_model(model, x_test, y_test)
    report.save_classification_report(classification_results, OUTPUT_DIR)

    # Save the plots
    report.plot_metrics(history, OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a VGG16 model on CIFAR10 using Keras"
    )
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)
