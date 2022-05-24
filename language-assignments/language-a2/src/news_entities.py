import pandas as pd
import spacy
from spacy.tokens import Doc
from pathlib import Path
from typing import Sequence, Callable, List

import src.geopol as geopol


def read_news(file_path: Path) -> pd.DataFrame:
    """Reads news df with correct index"""
    return pd.read_csv(file_path, index_col=0).reset_index(drop=True)


def list_sentiment(docs: Sequence[Doc], sent_f: Callable[[Doc], float]) -> List[float]:
    return [sent_f(doc) for doc in docs]


def df_ent_and_sent(df: pd.DataFrame, nlp: spacy.lang, sent_fun) -> pd.DataFrame:
    """Full NLP pipeline (entity extraction and sentiment)"""
    headline_docs = list(nlp.pipe(df["title"]))
    geopols = geopol.list_geopol(headline_docs)
    sentiments = list_sentiment(headline_docs, sent_fun)
    return pd.DataFrame(
        zip(df["title"], geopols, sentiments), columns=["title", "GPE", "sentiment"]
    )

