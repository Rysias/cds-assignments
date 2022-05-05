from sklearn.linear_model import LogisticRegression
import src.load_data as load_data
import src.report_performance as rp
import argparse
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main(args):
    dataset = args.dataset
    (x_train, y_train, x_test, y_test) = load_data.load_dataset(dataset)
    logging.info("Data loaded!")

    logging.info("Training model...")
    logistic = LogisticRegression()
    logistic.fit(x_train, y_train)
    logging.info("Model trained!")

    report = rp.get_classification_report(logistic, x_test, y_test)
    logging.info("Model evaluated! Report:\n{}".format(report))
    rp.save_classification_report(report, "lr_report")


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
