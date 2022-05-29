import argparse
import logging
import pandas as pd
import src.augment as augment
from pathlib import Path


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_more_train(train_path: Path = Path("input/train.csv")) -> pd.DataFrame:
    """
    Loads the train data and returns the dataframe with the additional training data
    """
    rawdf = pd.read_csv(train_path)
    df = rawdf.loc[rawdf["threat"] == 1, ["threat", "comment_text"]]
    df.columns = ["label", "text"]
    return df


def main(args) -> None:
    datapath = Path(args.dataset)
    if not datapath.exists():
        logging.error(f"The dataset path {datapath} does not exist.")
        exit(1)

    # split dataset into train and test
    df = pd.read_csv(datapath)
    df_train = df.sample(frac=0.8, random_state=42)
    df_test = df.drop(df_train.index)
    # save test
    df_test.to_csv(Path("input/test.csv"), index=False)

    toxicdf = df_train[df_train["label"] == 1]
    nontoxicdf = df_train[df_train["label"] == 0]
    # Add additional training data
    logging.info("Loading more training data")
    toxicdf = pd.concat([toxicdf, get_more_train()])
    logging.info(f"Augmenting toxic data...")
    augdf = augment.synonym_augment(toxicdf, n=args.augment_size)
    logging.info("Done! writing to file...")
    full_df = (
        pd.concat([augdf, toxicdf, nontoxicdf]).drop_duplicates().reset_index(drop=True)
    )
    full_df.to_csv(Path("input/augmented_train_data.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates training and test set for DNN classifier via extra data and data augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="input/VideoCommentsThreatCorpus.csv",
        help="Path to dataset",
    )
    parser.add_argument(
        "--augment-size",
        type=int,
        default=3,
        help="Number of times to augment each row",
    )
    args = parser.parse_args()
    main(args)
