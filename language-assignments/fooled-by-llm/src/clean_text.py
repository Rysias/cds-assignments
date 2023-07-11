"""
Functions for doing rudimentary cleaning of text in news articles
"""
import pandas as pd


def remove_tags(text: pd.Series) -> pd.Series:
    """Removes (description) tags from text"""
    new_text = text.str.replace(r"\(.*\)", "", regex=True)
    new_text = new_text.str.replace(r"\[.*\]", "", regex=True)
    return new_text


def remove_url(text: pd.Series) -> pd.Series:
    """Removes url from text"""
    return text.str.replace(r"http\S+", "", regex=True)


def clean_text(text: pd.Series) -> pd.Series:
    """
    Cleans text
    """
    text = remove_tags(text)
    text = remove_url(text)
    text = text.str.replace(r"\n", " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True)
    text = text.str.strip()
    return text


def remove_text_to_dash(text: pd.Series) -> pd.Series:
    """Removes text before the first dash"""
    new_text = text.str.replace(r"^[^-]*-", "", regex=True).str.strip()
    return new_text


def clean_true_news(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        short_text=remove_text_to_dash(df["short_text"]).where(
            df["type"] == "True", df["short_text"]
        )
    )
