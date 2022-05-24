import pytest
import spacy

import src.textblob as textblob
import src.news_entities as ne


@pytest.fixture(scope="session")
def nlp():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("spacytextblob")
    return nlp


def test_textblob(nlp):
    doc = nlp("you are stupid and dumb :(")
    assert textblob.textblob_sentiment(doc) < 0


def test_multiple_sentiment(nlp):
    docs = list(nlp.pipe(["I am angry!", "Happy days people"]))
    sentiments = ne.list_sentiment(docs, sent_f=textblob.textblob_sentiment)
    assert sentiments[0] < 0
    assert sentiments[1] > 0


def test_initialize_textblob():
    new_nlp = spacy.load("en_core_web_sm")
    sentiment_function = textblob.initialise_textblob(new_nlp)
    assert new_nlp.has_pipe("spacytextblob")
    assert callable(sentiment_function)
