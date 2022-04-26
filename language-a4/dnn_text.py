"""
Trains a deep neural network for detecting toxicity
"""
import argparse
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(Path(args.dataset))

    model_path = "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2"
    hub_layer = hub.KerasLayer(
        model_path, input_shape=[], dtype=tf.string, trainable=False
    )

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(1))

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

