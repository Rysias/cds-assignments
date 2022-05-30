"""
Cleans the gpt-generated texts
"""
import pandas
from pathlib import Path
import src.clean_text as clean_text
import argparse


def main(args: argparse.Namespace) -> None:
    MODEL_NAME = args.model_name
    input_path = Path(f"data/{MODEL_NAME}_prompts.csv")
    df = pandas.read_csv(input_path)
    clean_df = df.copy()
    clean_df["generated_text"] = clean_text.clean_text(clean_df["generated_text"])
    output_path = Path(f"data/{input_path.stem}_clean.csv")
    clean_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean prompts")
    parser.add_argument("--model-name", "-m", type=str, default="gpt-neo-125m")
    args = parser.parse_args()
    main(args)
