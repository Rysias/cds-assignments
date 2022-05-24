"""
Create a program which allows a user to define a number of different collocates at the same time, rather than only one
"""
import src.collocate as clt
import argparse
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def join_texts(file_list):
    return " ".join(clt.clean_file(file_path) for file_path in file_list)


def process_corpus(file_list):
    logging.info("processing corpus...")
    joined_text = join_texts(file_list)
    return clt.tokenize_doc(joined_text)


def main(args):
    search_terms = args.search_terms
    window_size = args.window_size
    data_dir = Path(args.data_dir)
    file_path = data_dir / args.file_name

    doc = clt.get_doc(file_path)
    corpus = clt.get_word_list(doc)

    colloc_list = [None for _ in range(len(search_terms))]
    for i, search_term in enumerate(search_terms):
        colloc_list[i] = clt.collocate_pipeline(
            corpus, search_term, window_size
        ).assign(node_word=search_term)

    pd.concat(colloc_list).to_csv(
        f"output/collocation_{file_path.stem}.csv", index=False
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Creates collocation info for a multiple search terms in a single file. The output is called collocation_{text_name}.csv"
    )
    argparser.add_argument(
        "--file-name",
        required=True,
        help="file-name to search through (in 100_english_novels",
    )
    argparser.add_argument(
        "--search-terms",
        nargs="+",
        required=True,
        help="Node words to find collocates (space delimited)",
    )
    argparser.add_argument(
        "--data-dir",
        default="input/",
        help="Directory, where texts are located",
    )
    argparser.add_argument(
        "--window-size",
        required=True,
        type=int,
        help="Window size (on each side of node word)",
    )
    args = argparser.parse_args()
    main(args)
