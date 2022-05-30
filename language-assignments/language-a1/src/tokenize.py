"""
Tokenizes text with spaCy.
"""

import src.util as util
import spacy
import numpy as np
from spacy.tokens import Doc
from pathlib import Path
from typing import Sequence

# Disable everything but tokenisation to improve performance
NLP = spacy.load(
    "en_core_web_sm",
    exclude=["tagger", "parser", "ner", "tok2vec", "attribute_ruler", "lemmatizer"],
)
NLP.max_length = 100000000  # For handling funky bug - too long to actually matter


def tokenize_doc(text, n_cores: int = 1) -> Doc:
    return next(NLP.pipe([text], n_process=n_cores, disable=NLP.pipe_names))


def tokenize_docs(texts, n_cores=1):
    return NLP.pipe(
        [text.lower() for text in texts], n_process=n_cores, disable=NLP.pipe_names
    )


def get_doc(file_path: Path) -> Doc:
    clean_text = util.clean_file(file_path)
    return tokenize_doc(clean_text)


def text_to_word_list(text_list: Sequence[Doc]) -> np.ndarray:
    return np.concatenate((get_word_list(doc) for doc in text_list), axis=None)


def get_word_list(doc: Doc) -> np.ndarray:
    return np.array([tok.text for tok in doc if not tok.is_punct | tok.is_space])
