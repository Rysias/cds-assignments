import pandas as pd

if __name__ == "__main__":
    rawdf = pd.read_csv("input/train.csv")
    df = rawdf[rawdf["toxic"] == 1, ["toxic", "comment_text"]]
    df.columns = ["label", "text"]
    df.to_csv("input/toxic_train.csv", index=False)
