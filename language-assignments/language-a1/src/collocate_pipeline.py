import src.tokenize as tokenize
import src.collocate as collocate
from pathlib import Path


def process_file(file_path: Path, search_term: str, window_size: int):
    doc = tokenize.get_doc(file_path)
    corpus = tokenize.get_word_list(doc)
    return collocate.collocate_pipeline(corpus, search_term, window_size)
