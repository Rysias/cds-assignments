from sklearn.linear_model import LogisticRegression
import src.load_data as load_data
import argparse
import numpy as np
from sklearn.metrics import classification_report
from pathlib import Path


def get_classification_report(
    model: LogisticRegression, x_test: np.ndarray, y_test: np.ndarray
) -> str:
    """
    Generates a classification report for the given model.
    """
    predictions = model.predict(x_test)
    return classification_report(y_test, predictions)


def save_classification_report(report: str, filename: Path) -> None:
    """ Save classification report as txt file """
    with open(filename, "w") as f:
        f.write(report)


def main(args):
    dataset = args.dataset
    (x_train, y_train, x_test, y_test) = load_data.load_dataset(dataset)

    logistic = LogisticRegression()
    logistic.fit(x_train, y_train)
    report = get_classification_report(logistic, x_test, y_test)
    print(report)
    save_classification_report(report, Path("out/lr_report.txt"))


if __name__ == "__main__":
    # add argparse for selecting dataset (choose between mnist and cifar10)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Choose to use either mnist or cifar10 (default: %(default)s)",
    )
    args = parser.parse_args()

    main(args)
