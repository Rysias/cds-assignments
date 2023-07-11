import src.load_data as load_data
from src.neuralnetwork import NeuralNetwork
import src.report_performance as rp
import argparse
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main(args):
    dataset = args.dataset
    epochs = args.epochs
    (x_train, y_train, x_test, y_test) = load_data.load_dataset(dataset)
    logging.info("Data loaded!")

    logging.info("initializing neural network...")
    layers = (x_train.shape[1], 64, 10)
    nn = NeuralNetwork(layers=layers)
    logging.info("training neural network...")
    nn.fit(x_train, y_train, epochs=epochs)
    logging.info("neural network trained!")

    report = rp.get_classification_report(nn, x_test, y_test)
    logging.info("Model evaluated! Report:\n{}".format(report))
    rp.save_classification_report(report, "nn_report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    # Add epochs argument
    parser.add_argument("--epochs", default=10, type=int)
    args = parser.parse_args()
    main(args)
