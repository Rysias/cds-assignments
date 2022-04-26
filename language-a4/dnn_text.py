"""
Trains a deep neural network for detecting toxicity
"""
import argparse
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def main(args: argparse.Namespace) -> None:

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

