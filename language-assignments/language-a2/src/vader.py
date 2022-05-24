import functools
from spacy.tokens import Doc  # type: ignore
from typing import Callable
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def vader_sentiment(doc: Doc, analyzer: SentimentIntensityAnalyzer) -> float:
    return analyzer.polarity_scores(doc.text)["compound"]


def initialise_vader() -> Callable[[Doc], float]:
    analyzer = SentimentIntensityAnalyzer()
    return functools.partial(vader_sentiment, analyzer=analyzer)
