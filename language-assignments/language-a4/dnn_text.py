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
    rawdf = pd.read_csv(Path(args.dataset))
    toxicdf = rawdf[rawdf["label"] == 1]
    non_toxicdf = rawdf[rawdf["label"] == 0]
    aug_df = augment.synonym_augment(toxicdf, "text", "label", n=10)
    df = pd.concat([toxicdf, non_toxicdf, aug_df])
    X_train, X_test, y_train, y_test = train_test_split(
        df[["text"]], df["label"], test_size=0.2
    )
    # sampler = RandomUnderSampler(random_state=42)
    # X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    assert X_train.shape[0] == y_train.shape[0]
    model = convnet.create_model(dropout=DROPOUT)
    logging.info(f"model summary: {model.summary()}")

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
        "--dataset",
        default="input/VideoCommentsThreatCorpus.csv",
        type=str,
        required=False,
        help="Path to the dataset.",
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

