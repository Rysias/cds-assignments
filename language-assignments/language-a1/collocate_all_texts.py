"""
- Create a program which does the above for every novel in the corpus, saving one output CSV per novel
"""
import src.collocate as clt
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def tokenize_docs(texts, n_cores=1):
    return clt.NLP.pipe(texts, n_process=n_cores, disable=clt.NLP.pipe_names)


def main(args):
    RUN_TEST = args.test_mode
    search_term = args.search_term
    window_size = args.window_size
    path_dir = Path(args.corpus_path)

    logging.info("Cleaning all texts...")
    all_texts = [
        (clt.clean_file(file), file)
        for i, file in enumerate(path_dir.glob("*.txt"))
        if i < 11
    ]
    logging.info("Tokenizing all texts...")
    all_docs = tokenize_docs(text for text, _ in all_texts)
    files = [file for _, file in all_texts]
    for i, text in enumerate(all_docs):
        corpus = clt.get_word_list(text)
        collate_df = clt.collocate_pipeline(corpus, search_term, window_size)
        clt.write_output(collate_df, files[i].name, search_term)
        if i % 10 == 0:
            logging.info(f"processed {i+1} of {len(all_texts)} files")

        if i == 10 and RUN_TEST:
            break


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Creates collocation info for all files in corpus. The output is written to a csv with the name '{text_name}_{search_string}.csv' in the output folder"
    )
    argparser.add_argument(
        "--test-mode", action="store_true", help="Runs a small test", default=False
    )
    argparser.add_argument(
        "--corpus-path",
        default="./input/",
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
