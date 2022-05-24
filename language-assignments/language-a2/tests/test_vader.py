from spacy.vocab import Vocab  # type: ignore
from spacy.tokens import Doc  # type: ignore
import pytest
import src.vader as vader


@pytest.fixture(scope="session")
def vocab():
    return Vocab(strings=["I am angry !".split()])


def test_initialise_vader(vocab):
    sentiment_function = vader.initialise_vader()
    doc = Doc(vocab, words=["I", "am", "angry", "!"])
    assert callable(sentiment_function)
    assert type(sentiment_function(doc)) is float

