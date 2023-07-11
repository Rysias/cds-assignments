"""
Cleans textual data for prediction tasks
"""
import pandas as pd
from pathlib import Path


def clean_text(text: pd.Series) -> pd.Series:
    """Remove whitespace and punctuation"""
    return (
        text.str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.rstrip()
    )


def remove_short_text(
    df: pd.DataFrame, textcol: str = "text", min_length: int = 10
) -> pd.DataFrame:
    """
    Remove text that is too short
    """
    text_filter = df[textcol].str.len() >= min_length
    return df[text_filter]


def clean_df(df: pd.DataFrame, textcol: str = "text") -> pd.DataFrame:
    """
    Cleans dataframe with text column
    """
    clean_df = df.copy()
    clean_df[textcol] = clean_text(df[textcol])
    return remove_short_text(clean_df, textcol)


def read_and_clean(filename: str) -> pd.DataFrame:
    """
    Reads and cleans dataframe from csv file
    """
    df = pd.read_csv(filename)
    return clean_df(df)
