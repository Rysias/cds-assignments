"""
Trains a deep neural network for detecting toxicity
"""
import argparse
import pandas as pd

# import tensorflow metrics

from imblearn.under_sampling import RandomUnderSampler  # type: ignore
from sklearn.model_selection import train_test_split
import src.report_performance as rp
import src.convnet as convnet
import src.augment as augment
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)


BATCH_SIZE = 32


def main(args: argparse.Namespace) -> None:
    EPOCHS = args.epochs
    DROPOUT = args.dropout
    df_train = pd.read_csv(Path(args.train_data))
    df_test = pd.read_csv(Path(args.test_data))
    X_train = df_train[["text"]].values
    y_train = df_train["label"].values
    X_test = df_test[["text"]].values
    y_test = df_test["label"].values

    assert X_train.shape[0] == y_train.shape[0]
    model = convnet.create_model(dropout=DROPOUT)
    logging.info(f"model summary: {model.summary()}")

    # undersample
    rus = RandomUnderSampler(random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    # Train the model
    history = model.fit(
        X_train,
        y_test,
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
        description="Trains a deep neural network for detecting toxicity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-data",
        default="input/augmented_train_data.csv",
        type=str,
        required=False,
        help="Path to the train dataset.",
    )
    parser.add_argument(
        "--test-data",
        default="input/test.csv",
        type=str,
        required=False,
        help="Path to the test dataset.",
    )
    # add epoch as argument
    parser.add_argument(
        "--epochs", default=1, type=int, required=False, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", default=32, type=int, required=False, help="Batch size"
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

