"""
Create a program which does this for the whole dataset, creating a CSV with one set of results, showing the mutual information scores for collocates across the whole set of texts
"""

import src.collocate as clt
import argparse
from pathlib import Path


def join_texts(file_list):
    return " ".join(clt.clean_file(file_path) for file_path in file_list)


def process_corpus(file_list):
    joined_text = join_texts(file_list)
    return clt.tokenize_doc(joined_text)


def main(args):
    search_term = args.search_term
    window_size = args.window_size
    path_dir = Path(args.corpus_path)

    all_paths = list(path_dir.glob("*.txt"))
    big_text = process_corpus(all_paths)

    corpus = clt.get_word_list(big_text)
    collate_df = clt.collocate_pipeline(corpus, search_term, window_size)

    collate_df.to_csv(Path("output") / f"corpus_{search_term}.csv")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Creates collocation info for the whole corpus. The script treats the corpus as a single file. The output is written to a csv with the name 'corpus_{search_string}.csv' in the output folder"
    )
    argparser.add_argument(
        "--corpus-path",
        required=True,
        help="full- or relative path to the folder to search through",
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
    args = argparser.parse_args()
    main(args)
