import argparse
import pandas as pd
import spacy
import numpy as np
import functools
import seaborn as sns
import matplotlib.pyplot as plt
from spacy.tokens import Doc
from pathlib import Path
from spacytextblob.spacytextblob import SpacyTextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Sequence, Callable, List, Tuple

NLP = spacy.load("en_core_web_sm")


def extract_geopol(doc: Doc) -> str:
    """ Extracts geopolitical entities from a Doc 
    Outputs a semicolon-separated stringg
    """
    return ";".join(ent.text for ent in doc.ents if ent.label_ == "GPE")


def initialise_textblob(spacy_model: spacy.lang) -> Callable[[Doc], float]:
    spacy_model.add_pipe("spacytextblob")
    return textblob_sentiment


def vader_sentiment(doc: Doc, analyzer: SentimentIntensityAnalyzer) -> float:
    return analyzer.polarity_scores(doc.text)["compound"]


def initialise_vader() -> Callable[[Doc], float]:
    analyzer = SentimentIntensityAnalyzer()
    return functools.partial(vader_sentiment, analyzer=analyzer)


def list_geopol(docs: Sequence[Doc]) -> List[str]:
    return [extract_geopol(doc) for doc in docs]


def textblob_sentiment(doc: Doc) -> float:
    return doc._.blob.polarity


def list_sentiment(docs: Sequence[Doc], sent_f: Callable[[Doc], float]) -> List[float]:
    return [sent_f(doc) for doc in docs]


def df_ent_and_sent(df: pd.DataFrame, sent_fun=textblob_sentiment) -> pd.DataFrame:
    """ Full NLP pipeline (entity extraction and sentiment) """
    headline_docs = list(NLP.pipe(df["title"]))
    geopols = list_geopol(headline_docs)
    sentiments = list_sentiment(headline_docs, sent_fun)
    return pd.DataFrame(
        zip(df["title"], geopols, sentiments), columns=["title", "GPE", "sentiment"]
    )


def flatten_series(series: pd.Series) -> pd.Series:
    return series.apply(pd.Series).stack().reset_index(drop=True)


def split_entities(ents: pd.Series) -> pd.Series:
    """ Transforms the semicolon seperated entities to one series """
    non_empty_ents = ents[ents.str.len() > 0]
    split_ents = non_empty_ents.str.split(";")
    return flatten_series(split_ents)


def most_common_ents(ents: pd.Series, n=20) -> pd.DataFrame:
    """ Formats and finds the most common entities """
    ent_series = split_entities(ents)
    return ent_series.value_counts()[:n].rename_axis("Entity").reset_index(name="Count")


def plot_top_ents(ent_df, output_dir, top_n=20, group_type="Real"):
    """ Finds the most common entities and olots them in a horizontal bar chart """
    top_ents = most_common_ents(ent_df["GPE"], n=top_n)
    plot_title = f"Most Mentioned {group_type} News GPEs"
    sns.barplot(data=top_ents, y="Entity", x="Count", orient="h", color="#29C5F6").set(
        title=plot_title
    )
    plt.savefig(str(output_dir / f"{group_type}_top_ents.png"))


def read_news(file_path: Path) -> pd.DataFrame:
    """ Reads news df with correct index """
    return pd.read_csv(file_path, index_col=0).reset_index(drop=True)


def create_output_dir() -> Path:
    output_dir = Path("output/")
    if not output_dir.exists():
        output_dir.mkdir()
    return output_dir

def process_news_df(df: pd.DataFrame, sentiment: Tuple[str, Callable], group_label: str) -> None:
    """ Full pipeline for extracting sentiment, GPEs, and creating plot """
    sent_name, sent_fun = sentiment
        
    # Calculate the stuff
    top_ents = df_ent_and_sent(df, sent_fun=sent_fun)
    
    # Plot
    output_dir = create_output_dir()
    plot_top_ents(top_ents, output_dir, group_type=group_label)

    # Write output
    top_ents.to_csv(output_dir / f"{group_label}_GPE_{sent_name}.csv")

def main(args):
    topn = args.top_n
    DATA_PATH = Path(args.data_path)

    if args.sentiment == "textblob":
        sentiment = ("textblob", initialise_textblob(NLP))
    elif args.sentiment == "vader":
        sentiment = ("vader", initialise_vader())

    news_df = read_news(DATA_PATH)
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
