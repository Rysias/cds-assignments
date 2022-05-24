"""
Creates a plot of the words with highest Mutual Information (MI) for collocates of node words
"""
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


def main(args: argparse.Namespace) -> None:
    """Main function"""
    # Read in data
    DATA_PATH = Path(args.file_path)
    rawdata = pd.read_csv(DATA_PATH)
    data = rawdata[rawdata["collocate_count"] > 1].nlargest(10, "MI")
    # Plot data
    sns.set(style="whitegrid")
    sns.set_context("poster")
    sns.barplot(x="MI", y="collocate", data=data, color="#5a7873").set(
        title="Bleak Collocates in 'Bleak House'"
    )
    plt.savefig(DATA_PATH.with_suffix(".png"), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the collocates of words with highest MI of 'bleak' in 'Bleak House' by Dickins."
    )
    parser.add_argument(
        "--file-path",
        type=str,
        required=True,
        help="Path to the collocate file for Bleak House.",
    )
    args = parser.parse_args()
    main(args)
