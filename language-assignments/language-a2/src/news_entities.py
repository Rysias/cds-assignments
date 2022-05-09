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


def extract_geopol(doc: Doc) -> str:
    """Extracts geopolitical entities from a Doc
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


def df_ent_and_sent(
    df: pd.DataFrame, nlp: spacy.lang, sent_fun=textblob_sentiment
) -> pd.DataFrame:
    """Full NLP pipeline (entity extraction and sentiment)"""
    headline_docs = list(nlp.pipe(df["title"]))
    geopols = list_geopol(headline_docs)
    sentiments = list_sentiment(headline_docs, sent_fun)
    return pd.DataFrame(
        zip(df["title"], geopols, sentiments), columns=["title", "GPE", "sentiment"]
    )


def flatten_series(series: pd.Series) -> pd.Series:
    return series.apply(pd.Series).stack().reset_index(drop=True)


def split_entities(ents: pd.Series) -> pd.Series:
    """Transforms the semicolon seperated entities to one series"""
    non_empty_ents = ents[ents.str.len() > 0]
    split_ents = non_empty_ents.str.split(";")
    return flatten_series(split_ents)


def most_common_ents(ents: pd.Series, n=20) -> pd.DataFrame:
    """Formats and finds the most common entities"""
    ent_series = split_entities(ents)
    return ent_series.value_counts()[:n].rename_axis("Entity").reset_index(name="Count")


def plot_top_ents(ent_df, output_dir, top_n=20, group_type="Real"):
    """Finds the most common entities and olots them in a horizontal bar chart"""
    top_ents = most_common_ents(ent_df["GPE"], n=top_n)
    plot_title = f"Most Mentioned {group_type} News GPEs"
    sns.barplot(data=top_ents, y="Entity", x="Count", orient="h", color="#29C5F6").set(
        title=plot_title
    )
    plt.savefig(str(output_dir / f"{group_type}_top_ents.png"))


def read_news(file_path: Path) -> pd.DataFrame:
    """Reads news df with correct index"""
    return pd.read_csv(file_path, index_col=0).reset_index(drop=True)
