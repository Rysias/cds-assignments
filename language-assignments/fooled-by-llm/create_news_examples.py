"""
script for creating a dataset of news-examples to be used for generating news-articles.
"""
from pathlib import Path
import pandas as pd
import argparse
import src.format_news as format_news

NEWS_TYPES = ["Fake", "True"]
DATA_DIR = Path("data/raw")
N = 40
N_SENT = 3
RANDOM_SEED = 1337


def main(args: argparse.Namespace) -> None:
    """
    main function for creating a dataset of news-examples to be used for generating news-articles.
    """
    N = args.n_articles
    N_SENT = args.n_sentences

    # create a sentencizer
    sentencizer = format_news.create_sentencizer()

    news_list = []
    # Load data
    for news_type in NEWS_TYPES:
        # load the data
        data = pd.read_csv(DATA_DIR / f"{news_type}.csv")

        # filter data to shorter than 75 words
        text_len = data["text"].str.split().str.len()
        len_crit = (3 < text_len) & (text_len < 75)
        short_data = data[len_crit]

        sample = short_data.sample(n=N, random_state=RANDOM_SEED)

        # create the news-examples
        sample["short_text"] = sample["text"].apply(
            lambda x: format_news.first_n_sentences(
                x, n=N_SENT, sentencizer=sentencizer
            )
        )
        sample["type"] = news_type

        news_list.append(sample)

    # concatenate the news-examples
    news_examples = pd.concat(news_list).reset_index(drop=True)
    # save the news-examples
    news_examples[["title", "subject", "date", "short_text", "type"]].to_csv(
        DATA_DIR.parent / "news_examples.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create news examples")
    parser.add_argument(
        "--n-articles", "-na", type=int, default=40, help="number of articles"
    )
    parser.add_argument(
        "--n-sentences", "-ns", type=int, default=3, help="number of sentences"
    )
    args = parser.parse_args()
    main(args)
