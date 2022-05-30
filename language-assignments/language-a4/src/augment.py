"""
Functions for doing NLP data augmentation
"""

import nlpaug.augmenter.word as naw  # type: ignore
import nltk
import pandas as pd
import numpy as np

aug = naw.SynonymAug(aug_src="wordnet", aug_max=2)


def handle_lookup(e: Exception, missing_import: str):
    if missing_import in str(e):
        nltk.download(missing_import)
        return True


def repeat_df(df: pd.DataFrame, n=1):
    return pd.DataFrame(np.repeat(df.to_numpy(), n, axis=0), columns=df.columns)


def handle_lookups(e: Exception):
    imports = ["wordnet", "averaged_perceptron_tagger", "omw-1.4"]
    for missing_import in imports:
        if handle_lookup(e, missing_import):
            return True
    raise e


def synonym_augment(
    df: pd.DataFrame, text_col: str = "text", label_col: str = "label", n: int = 2
) -> pd.DataFrame:
    """ Augments the data with n copies of each entry in the text column """
    try:
        # duplicate df
        aug_df = repeat_df(df, n)
        text = aug.augment(aug_df[text_col].tolist())
        return pd.DataFrame({"label": aug_df[label_col], "text": text,})
    except LookupError as e:
        handle_lookups(e)
        return synonym_augment(df, text_col=text_col, label_col=label_col, n=n)

