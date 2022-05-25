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
    Generates a prompt using OpenAI's GPT-2 model.
    """
    return openai.Completion.create(
        prompt=prompt,
        engine=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
    )["choices"][0]["text"]


def main(args: argparse.Namespace) -> None:
    MODEL_NAME = args.model_name
    df = pd.read_csv(Path("data/clean_news_examples.csv"))
    df["prompt"] = df["type"].str.lower() + " headline: " + df["title"] + "\n" + "text:"

    logging.info("Authenticating...")
    prompts.authenticate_goose(Path("config.json"))
    logging.info("generating prompts...")
    df["generated_text"] = df["prompt"].apply(
        lambda x: generate_prompt(x, model_name=MODEL_NAME)
    )
    logging.info("Done! Writing data...")
    output_path = Path(f"data/{MODEL_NAME}_prompts.csv")
    df.to_csv(output_path, index=False)
    return


if __name__ == "__main__":
    config = util.read_json(Path("model_options.json"))
    parser = argparse.ArgumentParser(description="Generate prompts for news examples")
    parser.add_argument(
        "--model-name", "-m", type=str, default="gpt-neo-125m", choices=config["models"]
    )
    args = parser.parse_args()
    main(args)
