from sklearn.linear_model import LogisticRegression
import src.load_data as load_data
import src.report_performance as rp
import argparse


def main(args):
    dataset = args.dataset
    (x_train, y_train, x_test, y_test) = load_data.load_dataset(dataset)

    logistic = LogisticRegression()
    logistic.fit(x_train, y_train)
    report = rp.get_classification_report(logistic, x_test, y_test)
    print(report)
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
