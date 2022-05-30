"""
Implements the textblob sentiment analysis algorithm.
"""
import spacy  # type: ignore
from spacy.tokens import Doc  # type: ignore
from typing import Callable
from spacytextblob.spacytextblob import SpacyTextBlob

def textblob_sentiment(doc: Doc) -> float:
    return doc._.blob.polarity


def initialise_textblob(spacy_model: spacy.lang) -> Callable[[Doc], float]:
    spacy_model.add_pipe("spacytextblob")
    return textblob_sentiment
