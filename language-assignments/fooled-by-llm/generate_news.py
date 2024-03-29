"""
Adds GPT-generated prompts to the news set
"""
import pandas as pd
import argparse
import src.prompts as prompts
import src.util as util
import logging
from pathlib import Path
from src.prompts import PROMPT_FUNCS

# add basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def main(args: argparse.Namespace) -> None:
    MODEL_NAME = args.model_name
    MAX_TOKENS = args.max_tokens
    TEMPERATURE = args.temperature
    PROMPT_FUNC = PROMPT_FUNCS[args.prompt_function]
    df = pd.read_csv(Path(args.file_path))
    df["prompt"] = PROMPT_FUNC(df)

    logging.info("Authenticating...")
    prompts.authenticate_goose(Path("config.json"))
    logging.info("generating prompts...")
    df["generated_text"] = df["prompt"].apply(
        lambda x: prompts.generate_prompt(
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
        description="Generate news using a GPT model from goose.ai",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="gpt-neo-125m",
        choices=config["models"],
        help="GPT model to use for generation",
    )
    parser.add_argument(
        "--prompt-function",
        "-p",
        type=str,
        default="type_title_prompt",
        choices=PROMPT_FUNCS.keys(),
        help="Function to use for generating prompts",
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
