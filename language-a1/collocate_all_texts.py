"""
- Create a program which does the above for every novel in the corpus, saving one output CSV per novel
"""
import collocate_single_text as clt
import argparse
from pathlib import Path


def main(args):
    search_term = args.search_term
    window_size = args.window_size
    path_dir = Path(args.corpus_path)
    
    all_texts = list(path_dir.glob('*.txt'))
    for i, text in enumerate(all_texts):
        collate_df = clt.process_file(text, search_term, window_size)
        clt.write_output(collate_df, text.name, search_term)
        if i % 10 == 0:
            print(f"processed {i+1} of {len(all_texts)} files")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description = "Creates collocation info for all files in corpus. The output is written to a csv with the name '{text_name}_{search_string}.csv' in the output folder")
    argparser.add_argument("--corpus-path", required=True, help="full- or relative path to the folder to search through")
    argparser.add_argument("--search-term", required=True, help="Node word to find collocates")
    argparser.add_argument("--window-size", required=True, type=int, help="Window size (on each side of node word)")
    args = argparser.parse_args()
    main(args)