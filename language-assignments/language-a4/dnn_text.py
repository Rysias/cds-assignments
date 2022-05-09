"""
Trains a deep neural network for detecting toxicity
"""
import argparse
import pandas as pd

# import tensorflow metrics

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import src.report_performance as rp
import src.convnet as convnet
from pathlib import Path


BATCH_SIZE = 64


def main(args: argparse.Namespace) -> None:
    EPOCHS = args.epochs
    DROPOUT = args.dropout
    df = pd.read_csv(Path(args.dataset))
    X_train, X_test, y_train, y_test = train_test_split(
        df[["text"]], df["label"], test_size=0.2
    )
    sampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    assert X_train.shape[0] == y_train.shape[0]
    model = convnet.create_model(dropout=DROPOUT)

    # Train the model
    history = model.fit(
        X_resampled,
        y_resampled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
    )

    # Evaluate the model
    report = rp.evaluate_model(model, X_test, y_test)
    rp.save_classification_report(report, "dnn_text")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a deep neural network for detecting toxicity"
    )
    parser.add_argument(
        "--dataset",
        default="in/VideoCommentsThreatCorpus.csv",
        type=str,
        required=False,
        help="Path to the dataset.",
    )
    # add epoch as argument
    parser.add_argument(
        "--epochs", default=1, type=int, required=False, help="Number of epochs"
    )
    parser.add_argument(
        "--dropout",
        default=0.5,
        type=float,
        required=False,
        help="Dropout rate (float between 0-1)",
    )

    args = parser.parse_args()
    main(args)

