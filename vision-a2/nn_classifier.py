import src.load_data as load_data
from src.neuralnetwork import NeuralNetwork
import src.report_performance as rp
import argparse


def main(args):
    dataset = args.dataset
    epochs = args.epochs
    (x_train, y_train, x_test, y_test) = load_data.load_dataset(dataset)

    layers = (x_train.shape[1], 64, 10)
    nn = NeuralNetwork(layers=layers)
    nn.fit(x_train, y_train, epochs=epochs)
    report = rp.get_classification_report(nn, x_test, y_test)
    print(report)
    rp.save_classification_report(report, "nn_report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    # Add epochs argument
    parser.add_argument("--epochs", default=10, type=int)
    args = parser.parse_args()
    main(args)

