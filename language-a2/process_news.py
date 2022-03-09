import spacy
import argparse
import pandas as pd
from pathlib import Path
from src import news_entities as ne
from typing import Sequence, Callable, List, Tuple


NLP = spacy.load("en_core_web_sm")


def create_output_dir() -> Path:
    output_dir = Path("output/")
    if not output_dir.exists():
        output_dir.mkdir()
    return output_dir

def process_news_df(df: pd.DataFrame, sentiment: Tuple[str, Callable], group_label: str) -> None:
    """ Full pipeline for extracting sentiment, GPEs, and creating plot """
    sent_name, sent_fun = sentiment
        
    # Calculate the stuff
    top_ents = ne.df_ent_and_sent(df, NLP, sent_fun=sent_fun)
    
    # Plot
    output_dir = create_output_dir()
    ne.plot_top_ents(top_ents, output_dir, group_type=group_label)

    # Write output
    top_ents.to_csv(output_dir / f"{group_label}_GPE_{sent_name}.csv")

def main(args):
    topn = args.top_n
    DATA_PATH = Path(args.data_path)

    if args.sentiment == "textblob":
        sentiment = ("textblob", ne.initialise_textblob(NLP))
    elif args.sentiment == "vader":
        sentiment = ("vader", ne.initialise_vader())

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
        default="../../../CDS-LANG/tabular_examples/fake_or_real_news.csv",
        help="Path to the fake_or_real_news.csv file",
    )
    args = argparser.parse_args()
    main(args)
