import pandas as pd
from pathlib import Path
import src.clean_text as clean_text

DATA_DIR = Path("data")


def main() -> None:
    data_path = DATA_DIR / "news_examples.csv"
    df = pd.read_csv(data_path)
    df["title"] = clean_text.clean_text(df["title"])
    clean_df = clean_text.clean_true_news(df)
    clean_path = DATA_DIR / "clean_news_examples.csv"
    clean_df.to_csv(clean_path, index=False)


if __name__ == "__main__":
    main()
