"""
Compares sentiment from the two different sentiment analysis algorithms across real and fake news.
"""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_sentiment_df(path: Path) -> pd.DataFrame:
    """ parses name of csv and adds it to the dataframe """
    df = pd.read_csv(path)
    file_info = path.stem.split("_")
    df["news_type"] = file_info[0]
    df["sentiment_model"] = file_info[2]
    return df


def main() -> None:
    DATA_DIR = Path("output/")
    all_dfs = list(DATA_DIR.glob("*_GPE_*.csv"))
    master_df = pd.concat([read_sentiment_df(f) for f in all_dfs])
    sns.barplot(x="news_type", y="sentiment", hue="sentiment_model", data=master_df).set(title="Comparison of Sentiment Algorithms")
    plt.savefig(DATA_DIR / "sentiment_comparison.png")


if __name__ == "__main__":
    main()

