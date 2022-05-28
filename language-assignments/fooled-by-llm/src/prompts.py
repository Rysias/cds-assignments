from pathlib import Path
import src.util as util
import openai
import pandas as pd


def get_api_key(config: Path, keyname: str = "goose_api") -> str:
    """
    Gets the API key from the config file.
    """
    return util.read_json(config)[keyname]


def authenticate_goose(config: Path) -> None:
    """
    Authenticates with the goose API.
    """
    api_key = get_api_key(config, keyname="goose_api")
    openai.api_key = api_key
    openai.api_base = "https://api.goose.ai/v1"


def type_title_prompt(
    df: pd.DataFrame, cat_col: str = "type", title_col: str = "title"
) -> pd.Series:
    """ Prepares a prompt for each news item based on type and title"""
    assert (
        cat_col in df.columns and title_col in df.columns
    ), f"{cat_col} or {title_col} not in df"
    return df[cat_col].str.lower() + " headline: " + df[title_col] + "\n" + "text:"


def type_title_date_prompt(
    df: pd.DataFrame,
    cat_col: str = "type",
    title_col: str = "title",
    date_col: str = "date",
) -> pd.Series:
    """ Prepares a prompt for each news item based on type and title"""
    assert (
        cat_col in df.columns and title_col in df.columns and date_col in df.columns
    ), f"{cat_col} or {title_col} or {date_col} not in df"
    return (
        df[cat_col].str.lower()
        + " headline ("
        + df[date_col]
        + "): "
        + df[title_col]
        + "\n"
        + "text:"
    )


PROMPT_FUNCS = {
    "type_title_prompt": type_title_prompt,
    "type_title_date_prompt": type_title_date_prompt,
}
