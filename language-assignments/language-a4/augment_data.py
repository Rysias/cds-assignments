import argparse
import logging
import pandas as pd
import src.augment as augment
from pathlib import Path


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(args) -> None:
    datapath = Path(args.dataset)
    if not datapath.exists():
        logging.error(f"The dataset path {datapath} does not exist.")
        exit(1)
    df = pd.read_csv(datapath)
    toxicdf = df[df["label"] == 1]
    nontoxicdf = df[df["label"] == 0]
    logging.info(f"Augmenting toxic data...")
    augdf = augment.synonym_augment(toxicdf, n=args.augment_size)

    full_df = (
        pd.concat([augdf, toxicdf, nontoxicdf]).drop_duplicates().reset_index(drop=True)
    )
    full_df.to_csv(Path("input/augmented_data.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augments data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="input/VideoCommentsThreatCorpus.csv",
        help="Path to dataset",
    )
    parser.add_argument(
        "--augment-size",
        type=int,
        default=10,
        help="Number of times to augment each row",
    )
    args = parser.parse_args()
    main(args)
