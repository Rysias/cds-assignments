import argparse
from sklearn.model_selection import train_test_split

import src.clean_data as cd
import src.logistic as logistic
import src.report_performance as rp


def main(args) -> None:
    df = cd.read_and_clean(args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        df[["text"]], df["label"], test_size=0.2
    )
    assert X_train.shape[0] == y_train.shape[0]
    pipeline = logistic.create_pipeline(textcol="text")
    pipeline.fit(X_train, y_train)
    report = rp.get_classification_report(pipeline, X_test, y_test)
    rp.save_classification_report(report, "logistic_regression")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Performs logistic regression on the given dataset."
    )
    parser.add_argument(
        "--dataset",
        default="input/VideoCommentsThreatCorpus.csv",
        type=str,
        required=False,
        help="Path to the dataset.",
    )
    args = parser.parse_args()

    main(args)

