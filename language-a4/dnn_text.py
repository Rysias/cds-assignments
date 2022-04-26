"""
Trains a deep neural network for detecting toxicity
"""
import argparse
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

# import tensorflow metrics
from tensorflow.keras import metrics

from sklearn.model_selection import train_test_split
import src.report_performance as rp
from pathlib import Path


BATCH_SIZE = 64


def compile_model(model: tf.keras.Model,) -> None:
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[metrics.AUC, metrics.Accuracy],
    )


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(Path(args.dataset))
    X_train, X_test, y_train, y_test = train_test_split(
        df[["text"]], df["label"], test_size=0.2
    )
    assert X_train.shape[0] == y_train.shape[0]

    model_path = "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2"
    hub_layer = hub.KerasLayer(
        model_path, input_shape=[], dtype=tf.string, trainable=False
    )

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))

    compile_model(model)

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=1,
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
    args = parser.parse_args()
    main(args)

