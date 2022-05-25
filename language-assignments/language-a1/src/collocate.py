import pandas as pd
import numpy as np
import logging
from collections import Counter
from typing import List, Iterable

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_window(words: np.ndarray, idx: int, window_size: int) -> np.ndarray:
    return np.concatenate(
        (words[idx - window_size : idx], words[idx + 1 : idx + window_size + 1]),
        axis=None,
    )


def flatten_list(lst: Iterable[Iterable]) -> List:
    return [x for sub in lst for x in sub]


def count_words(corpus: Iterable, term: str) -> int:
    return np.sum(corpus == term)


def find_target_idx(words: Iterable[str], search_term: str):
    return [i for i, word in enumerate(words) if word == search_term]


def find_collocates(corpus, search_term_idx, window_size):
    all_collocates = [get_window(corpus, idx, window_size) for idx in search_term_idx]
    return flatten_list(all_collocates)


def create_collocate_df(corpus, search_term, window_size):
    search_term_idx = find_target_idx(corpus, search_term)
    all_collocates = find_collocates(corpus, search_term_idx, window_size)
    collocate_counts = Counter(all_collocates)
    return (
        pd.DataFrame.from_dict(collocate_counts, orient="index")
        .reset_index()
        .rename(columns={"index": "collocate", 0: "collocate_count"})
    )


def calc_mi(collocate_count, corpus_freq, corpus, search_term, window_size):
    corpus_size = len(corpus)
    node_freq = count_words(corpus, search_term)
    return np.log10(
        (collocate_count * corpus_size) / (node_freq * corpus_freq * window_size * 2)
    ) / np.log10(2)


def collocate_pipeline(corpus, search_term, window_size):
    df = create_collocate_df(corpus, search_term, window_size)
    logging.info(f"Created collocate df for {search_term}")
    df["corpus_count"] = df["collocate"].apply(lambda x: count_words(corpus, x))
    logging.info(f"Calculated corpus count for {search_term}")
    df["MI"] = calc_mi(
        df["collocate_count"], df["corpus_count"], corpus, search_term, window_size
    )
    logging.info(f"Calculated MI for {search_term}")
    return df

