"""
- Take a user-defined search term and a user-defined window size.
- Take one specific text which the user can define.
- Find all the context words which appear ± the window size from the search term in that text.
- Calculate the mutual information score for each context word.
- Save the results as a CSV file with (at least) the following columns: the collocate term; how often it appears as a collocate; how often it appears in the text; the mutual information score.
"""
from spacy.tokens import Doc
from typing import Iterable, List, Sequence
from spacy import load
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from spacy.tokens import Doc
from typing import List, Sequence

NLP = load(
    "en_core_web_sm",
    exclude=["tagger", "parser", "ner", "tok2vec", "attribute_ruler", "lemmatizer"],
)
NLP.max_length = 100000000


def read_txt(file_path: Path) -> List[str]:
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return f.read().splitlines()


def get_window(words: np.ndarray, idx: int, window_size: int) -> np.ndarray:
    return np.concatenate(
        (words[idx - window_size : idx], words[idx + 1 : idx + window_size + 1]),
        axis=None,
    )


def flatten_list(lst: Iterable[Iterable]):
    return [x for sub in lst for x in sub]


def count_words(corpus: Iterable, term: str) -> int:
    return np.sum(corpus == term)


def clean_file(file_path: Path) -> str:
    raw_text = read_txt(file_path)
    return " ".join(raw_text).lower()


def tokenize_doc(text, n_cores: int = 1) -> Doc:
    return next(NLP.pipe([text], n_process=n_cores, disable=NLP.pipe_names))


def tokenize_docs(texts, n_cores=1):
    return NLP.pipe(
        [text.lower() for text in texts], n_process=n_cores, disable=NLP.pipe_names
    )


def get_doc(file_path: Path) -> Doc:
    clean_text = clean_file(file_path)
    return tokenize_doc(clean_text)


def text_to_word_list(text_list: Sequence[Doc]) -> np.ndarray:
    return np.concatenate((get_word_list(doc) for doc in text_list), axis=None)


def get_word_list(doc: Doc) -> np.ndarray:
    return np.array([tok.text for tok in doc if not tok.is_punct | tok.is_space])


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
    df["corpus_count"] = df["collocate"].apply(lambda x: count_words(corpus, x))
    df["MI"] = calc_mi(
        df["collocate_count"], df["corpus_count"], corpus, search_term, window_size
    )
    return df


def process_file(file_path, search_term, window_size):
    doc = get_doc(file_path)
    corpus = get_word_list(doc)
    return collocate_pipeline(corpus, search_term, window_size)


def process_file2(file_path, search_term, window_size):
    doc = read_txt(file_path)
    corpus = text_to_word_list(doc)
    return collocate_pipeline(corpus, search_term, window_size)


def write_output(collocate_df, file_name, search_term):
    output_name = Path("output") / f"{file_name[:-4]}_{search_term}.csv"
    collocate_df.to_csv(output_name, index=False)


def main(args):
    DATA_DIR = Path(args.data_dir)
    search_term = args.search_term
    window_size = args.window_size
    file_path = DATA_DIR / args.file_name

    collocate_df = process_file(file_path, search_term, window_size)
    write_output(collocate_df, args.file_name, search_term)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Creates collocation info for a given file, search term, and window size. The output is written to a csv with the name '{text_name}_{search_string}.csv' in the output folder"
    )
    argparser.add_argument(
        "--file-name",
        required=True,
        help="file-name to search through (in 100_english_novels",
    )
    argparser.add_argument(
        "--search-term", required=True, help="Node word to find collocates"
    )
    argparser.add_argument(
        "--window-size",
        required=True,
        type=int,
        help="Window size (on each side of node word)",
    )
    argparser.add_argument(
        "--data-dir",
        default="../../../CDS-LANG/100_english_novels/corpus",
        help="Where to look for texts",
    )
    args = argparser.parse_args()
    main(args)
