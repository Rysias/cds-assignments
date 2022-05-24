import pandas as pd
from spacy.tokens import Doc  # type: ignore
from typing import Sequence, List


def extract_geopol(doc: Doc) -> str:
    """Extracts geopolitical entities from a Doc
    Outputs a semicolon-separated stringg
    """
    return ";".join(ent.text for ent in doc.ents if ent.label_ == "GPE")


def list_geopol(docs: Sequence[Doc]) -> List[str]:
    return [extract_geopol(doc) for doc in docs]


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
