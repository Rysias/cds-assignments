"""
- Take a user-defined search term and a user-defined window size.
- Take one specific text which the user can define.
- Find all the context words which appear ± the window size from the search term in that text.
- Calculate the mutual information score for each context word.
- Save the results as a CSV file with (at least) the following columns: the collocate term; how often it appears as a collocate; how often it appears in the text; the mutual information score.
"""
import argparse
from pathlib import Path
import src.collocate as collocate


def main(args):
    DATA_DIR = Path(args.data_dir)
    search_term = args.search_term
    window_size = args.window_size
    file_path = DATA_DIR / args.file_name

    collocate_df = collocate.process_file(file_path, search_term, window_size)
    collocate.write_output(collocate_df, args.file_name, search_term)


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
