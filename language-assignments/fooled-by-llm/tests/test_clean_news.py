import pandas as pd
import src.clean_text as clean_text


def test_clean_to_dash():
    text = pd.Series(
        [
            "text before dash - text after dash",
            "WASHINGTON (Reuters) - Donald Trump will speak with Hou",
        ]
    )
    new_text = clean_text.remove_text_to_dash(text)
    assert new_text.iloc[0] == "text after dash"
    assert new_text.iloc[1] == "Donald Trump will speak with Hou"


def test_clean_true_news():
    data = pd.DataFrame(
        {
            "type": ["True", "Fake", "True"],
            "short_text": [
                "text before dash - text after dash",
                "NEW YORK (Reuters) - Trump says he will be the next",
                "WASHINGTON (Reuters) - Donald Trump will speak at closed-door meeting",
            ],
        }
    )
    clean_data = clean_text.clean_true_news(data)
    assert clean_data["short_text"].iloc[0] == "text after dash"
    assert clean_data["short_text"].iloc[1] == data["short_text"].iloc[1]
    assert (
        clean_data["short_text"].iloc[2]
        == "Donald Trump will speak at closed-door meeting"
    )
