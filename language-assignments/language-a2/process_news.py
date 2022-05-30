"""
Main function for running the full pipeline
"""

import spacy  # type: ignore
import argparse
import pandas as pd
import logging
from pathlib import Path
from src import news_entities as ne
import src.plot_geopol as plot_geopol
import src.geopol as geopol
from typing import Callable, Tuple

# Add basic logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

NLP = spacy.load("en_core_web_sm")


def create_output_dir() -> Path:
    output_dir = Path("output/")
    if not output_dir.exists():
        output_dir.mkdir()
    return output_dir


def plot_top_ents(
    ents: pd.DataFrame, output_dir: Path, group_type: str, top_n: int = 10
) -> None:
    top_ents = geopol.most_common_ents(ents["GPE"], n=top_n)
    plot_geopol.plot_top_ents(top_ents, output_dir, group_type=group_type)


def process_news_df(
    df: pd.DataFrame, sentiment: Tuple[str, Callable], group_label: str
) -> None:
    """Full pipeline for extracting sentiment, GPEs, and creating plot"""
    sent_name, sent_fun = sentiment

    # Calculate the stuff
    logging.info(f"Calculating {sent_name} for {group_label} news")
    all_ents = ne.df_ent_and_sent(df, NLP, sent_fun=sent_fun)

    # Plot
    logging.info("plotting...")
    output_dir = create_output_dir()
    plot_top_ents(all_ents, output_dir, group_type=group_label)

    # Write output
    logging.info("writing output...")
    all_ents.to_csv(output_dir / f"{group_label}_GPE_{sent_name}.csv")


def main(args):
    DATA_PATH = Path(args.data_path)
    if args.sentiment == "textblob":
        import src.textblob as textblob

        sentiment = ("textblob", textblob.initialise_textblob(NLP))
    elif args.sentiment == "vader":
        import src.vader as vader

        sentiment = ("vader", vader.initialise_vader())
    else:
        raise ValueError("Invalid sentiment type")

    news_df = ne.read_news(DATA_PATH)
    mask = news_df["label"] == "REAL"
    real_df = news_df[mask]
    fake_df = news_df[~mask]

    process_news_df(real_df, sentiment, group_label="Real")
    process_news_df(fake_df, sentiment, group_label="Fake")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Finds Geopolitical entities (GEOP) and their sentiment from news headlines from a dataset containing fake and real news. Also plots the top n (default 20) most mentioned entities"
    )

    argparser.add_argument(
        "--sentiment",
        default="textblob",
        choices=["textblob", "vader"],
        help="Sentiment model to use (default: %(default)s)",
    )
    argparser.add_argument(
        "--top-n",
        default=20,
        type=int,
        help="Number of entities to plot (default: %(default)s)",
    )
    argparser.add_argument(
        "--data-path",
        default="input/fake_or_real_news.csv",
        help="Path to the fake_or_real_news.csv file",
    )
    args = argparser.parse_args()
    main(args)
