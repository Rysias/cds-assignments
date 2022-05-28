"""
Adds GPT-generated prompts to the news set
"""
import pandas as pd
import argparse
import src.prompts as prompts
import src.util as util
import openai
import logging
from pathlib import Path

# add basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def generate_prompt(
    prompt: str, model_name: str = "gpt-neo-125m", max_tokens: int = 75, temperature=0.9
) -> str:
    """
    Generates a prompt using a model from EleutherAI.
    """
    return openai.Completion.create(
        prompt=prompt,
        engine=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
    )["choices"][0]["text"]


def prepare_prompt(
    df: pd.DataFrame, cat_col: str = "type", title_col: str = "title"
) -> pd.Series:
    """ Prepares a prompt for each news item """
    assert (
        cat_col in df.columns and title_col in df.columns
    ), f"{cat_col} or {title_col} not in df"
    return df[cat_col].str.lower() + " headline: " + df[title_col] + "\n" + "text:"


def main(args: argparse.Namespace) -> None:
    MODEL_NAME = args.model_name
    MAX_TOKENS = args.max_tokens
    TEMPERATURE = args.temperature
    df = pd.read_csv(Path(args.file_path))
    df["prompt"] = prepare_prompt(df)

    logging.info("Authenticating...")
    prompts.authenticate_goose(Path("config.json"))
    logging.info("generating prompts...")
    df["generated_text"] = df["prompt"].apply(
        lambda x: generate_prompt(
            x, model_name=MODEL_NAME, max_tokens=MAX_TOKENS, temperature=TEMPERATURE
        )
    )
    logging.info("Done! Writing data...")
    output_path = Path(f"data/{MODEL_NAME}_prompts.csv")
    df.to_csv(output_path, index=False)
    return


if __name__ == "__main__":
    config = util.read_json(Path("model_options.json"))
    parser = argparse.ArgumentParser(
        description="Generate prompts for news examples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name", "-m", type=str, default="gpt-neo-125m", choices=config["models"]
    )
    parser.add_argument(
        "--file-path",
        "-f",
        type=str,
        default="data/clean_news_examples.csv",
        help="Path to the source for generating prompts",
    )
    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=75,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", "-T", type=float, default=0.9, help="Temperature for GPT model"
    )
    args = parser.parse_args()
    main(args)
