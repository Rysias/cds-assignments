"""
Splits the dataset randomly to get prompts and true
"""
import pandas as pd
import argparse
from pathlib import Path

RANDOM_SEED = 555


def main(args: argparse.Namespace) -> None:
    MODEL_NAME = args.model_name
    df = pd.read_csv(Path(f"data/{MODEL_NAME}_prompts_clean.csv"))

    # shuffle the data
    df = df.sample(frac=1, random_state=RANDOM_SEED)
    df.drop(columns=["prompt", "subject"], inplace=True)
    # create row id
    df["row_id"] = df.index

    # pivot generated_text adn short_text
    long_df = df.melt(
        id_vars=["title", "row_id", "date", "type"],
        value_vars=["short_text", "generated_text"],
        var_name="text_type",
        value_name="text",
    )
    long_df.sort_values(by=["row_id"], inplace=True)
    long_df.reset_index(drop=True, inplace=True)
    long_df["new_id"] = long_df.index

    dat1_filter = long_df["new_id"] % 2 == 1
    dat2_filter = ~dat1_filter

    dat1 = long_df[dat1_filter]
    dat2 = long_df[dat2_filter]

    df_len = df.shape[0]
    assert len(dat1) == len(dat2)
    assert dat1["text_type"].nunique() == 2
    assert ((df_len / 2) + 3) > (dat1["type"] == "Fake").sum() > ((df_len / 2) - 3)

    col_choices = ["row_id", "type", "title", "date", "text", "text_type"]
    dat1.to_csv(
        Path(f"data/{MODEL_NAME}_promptsv1.csv"), columns=col_choices, index=False
    )
    dat2.to_csv(
        Path(f"data/{MODEL_NAME}_promptsv2.csv"), columns=col_choices, index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean prompts")
    parser.add_argument("--model-name", "-m", type=str, default="gpt-neo-125m")
    args = parser.parse_args()
    main(args)
