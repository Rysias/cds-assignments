import pytest
import src.format_news as format_news


@pytest.fixture(scope="session")
def sentencizer():
    return format_news.create_sentencizer()


def test_first_n_sentences(sentencizer):
    this_sentencizer = sentencizer
    text = "This is a sentence. This is another sentence. This is the last sentence."
    assert (
        format_news.first_n_sentences(text, 2, this_sentencizer)
        == "This is a sentence. This is another sentence."
    )
    assert (
        format_news.first_n_sentences(text, 1, this_sentencizer)
        == "This is a sentence."
    )


def test_first_n_empty(sentencizer):
    this_sentencizer = sentencizer
    text = "this is the only sentence"
    assert format_news.first_n_sentences(text, 2, this_sentencizer) == text
