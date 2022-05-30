"""
Implements a logistic regression for classification of imbalanced data. 
"""
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

# import columntransformer from sklearn
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def create_pipeline(textcol: str = "clean_text") -> Pipeline:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    sampler = RandomUnderSampler(random_state=1)
    logistic = LogisticRegression()
    return Pipeline(
        [
            ("resample", sampler),
            (
                "column_transformer",
                ColumnTransformer(
                    [("comment_text_vectorizer", vectorizer, textcol)],
                    remainder="passthrough",
                ),
            ),
            ("model", logistic),
        ]
    )

