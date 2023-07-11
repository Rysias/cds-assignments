"""
Helper functions for formatting the news for Turing test.
"""
import spacy  # type: ignore


def create_sentencizer() -> spacy.pipeline.Sentencizer:
    """
    Creates a sentencizer pipe.
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.select_pipes(
        disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
    )
    nlp.add_pipe("sentencizer")
    return nlp


def first_n_sentences(text: str, n: int, sentencizer) -> str:
    """
    Returns the first n sentences of a text.
    """
    doc = sentencizer(text)
    return " ".join(str(sent) for i, sent in enumerate(doc.sents) if i < n)
