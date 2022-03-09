import pytest
import spacy
import pandas as pd

from src import news_entities as ne


@pytest.fixture(scope="session")
def nlp():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("spacytextblob")
    return nlp


def test_extract_geopol(nlp):
    doc = nlp("Washington battling Russia")
    geopols = ne.extract_geopol(doc)
    assert geopols == "Washington;Russia"


def test_only_GPE(nlp):
    doc = nlp("Why isn't Rihanna leading Denmark yet?")
    geopols = ne.extract_geopol(doc)
    assert geopols == "Denmark"


def test_no_GPE(nlp):
    doc = nlp("what?")
    geopols = ne.extract_geopol(doc)
    assert geopols == ""


def test_list_geopol(nlp):
    docs = list(nlp.pipe(["Denmark is a country", "Hello mr. smartypants"]))
    entities = ne.list_geopol(docs)
    assert entities[0] == "Denmark"
    assert entities[1] == ""


def test_textblob(nlp):
    doc = nlp("you are stupid and dumb :(")
    assert ne.textblob_sentiment(doc) < 0


def test_multiple_sentiment(nlp):
    docs = list(nlp.pipe(["I am angry!", "Happy days people"]))
    sentiments = ne.list_sentiment(docs, sent_f=ne.textblob_sentiment)
    assert sentiments[0] < 0
    assert sentiments[1] > 0


def test_split_entities():
    ents = pd.Series(["Washington;Denmark", "", "United States"])
    ent_list = ne.split_entities(ents)
    assert len(ent_list) == 3
    assert ent_list[0] == "Washington"
    assert ent_list[1] == "Denmark"
    assert all(len(ent) > 0 for ent in ent_list)


def test_initialise_vader(nlp):
    sentiment_function = ne.initialise_vader()
    doc = nlp("I am angry")
    assert callable(sentiment_function)
    assert type(sentiment_function(doc)) is float


def test_initialize_textblob():
    new_nlp = spacy.load("en_core_web_sm")
    sentiment_function = ne.initialise_textblob(new_nlp)
    assert new_nlp.has_pipe("spacytextblob")
    assert callable(sentiment_function)
