import src.geopol as geopol
import pandas as pd


def test_extract_geopol(nlp):
    doc = nlp("Washington battling Russia")
    geopols = geopol.extract_geopol(doc)
    assert geopols == "Washington;Russia"


def test_only_GPE(nlp):
    doc = nlp("Why isn't Rihanna leading Denmark yet?")
    geopols = geopol.extract_geopol(doc)
    assert geopols == "Denmark"


def test_no_GPE(nlp):
    doc = nlp("what?")
    geopols = geopol.extract_geopol(doc)
    assert geopols == ""


def test_list_geopol(nlp):
    docs = list(nlp.pipe(["Denmark is a country", "Hello mr. smartypants"]))
    entities = geopol.list_geopol(docs)
    assert entities[0] == "Denmark"
    assert entities[1] == ""


def test_split_entities():
    ents = pd.Series(["Washington;Denmark", "", "United States"])
    ent_list = geopol.split_entities(ents)
    assert len(ent_list) == 3
    assert ent_list[0] == "Washington"
    assert ent_list[1] == "Denmark"
    assert all(len(ent) > 0 for ent in ent_list)
