"""
Take a user-defined search term and a user-defined window size.
Take one specific text which the user can define.
Find all the context words which appear ± the window size from the search term in that text.
Calculate the mutual information score for each context word.
Save the results as a CSV file with (at least) the following columns: the collocate term; how often it appears as a collocate; how often it appears in the text; the mutual information score.
"""
import argparse
import logging
from pathlib import Path
import src.tokenize as tokenize
import src.collocate as collocate
import src.util as util

# add basic logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def process_file(file_path: Path, search_term: str, window_size: int):
    doc = tokenize.get_doc(file_path)
    corpus = tokenize.get_word_list(doc)
    return collocate.collocate_pipeline(corpus, search_term, window_size)


def main(args):
    DATA_DIR = Path(args.data_dir)
    search_term = args.search_term
    window_size = args.window_size
    file_path = DATA_DIR / args.file_name

    logging.info(f"Searching for {search_term} in {file_path}...")
    collocate_df = process_file(file_path, search_term, window_size)
    logging.info(f"writing output...")
    util.write_output(collocate_df, file_path, search_term)
    logging.info("done!")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Creates collocation info for a given file, search term, and window size. The output is written to a csv with the name '{text_name}_{search_string}.csv' in the output folder"
    )
    argparser.add_argument(
        "--file-name",
        required=True,
        help="file-name to search through (in 100_english_novels)",
    )
    argparser.add_argument(
        "--search-term", required=True, help="Node word to find collocates"
    )
    argparser.add_argument(
        "--window-size",
        default=6,
        type=int,
        help="Window size (on each side of node word)",
    )
    argparser.add_argument(
        "--data-dir",
        default="input/",
        help="Where to look for texts",
    )
    args = argparser.parse_args()
    main(args)
