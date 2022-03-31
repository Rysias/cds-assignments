from re import A
import src.load_data as load_data
from src.neuralnetwork import NeuralNetwork
import argparse


def main(args):
    dataset = args.dataset
    (x_train, y_train), (x_test, y_test) = load_data.load_dataset(dataset)
    nn = NeuralNetwork(layers=2)
    nn.fit(x_train, y_train, epochs=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    args = parser.parse_args()

