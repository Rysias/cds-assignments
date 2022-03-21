import argparse
import pandas as pd
import networkx as nx  # type: ignore
from functools import partial
from pathlib import Path
from src import calc_measures as cm
from src import visualize


def read_tsv(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath, sep="\t")


def write_measures(measure_df: pd.DataFrame, filepath: Path) -> None:
    """Writes measures to dataframe """
    output_path = Path("output") / f"{filepath.stem}_measures.csv"
    measure_df.to_csv(output_path, index=False)


def right_columns(df: pd.DataFrame) -> bool:
    cols = ["Source", "Target", "Weight"]
    return all(col in df.columns for col in cols)


def process_dir(dir_path: Path) -> None:
    """Processes all valid csv files in folder"""
    for filepath in dir_path.glob("*.csv"):
        process_file(filepath)


def process_file(filepath):
    """"Full pipeline for processing a (valid) file"""
    df = read_tsv(filepath)
    if not right_columns(df):
        return
    measure_dict = {
        "degree_centrality": nx.degree_centrality,
        "betweenness_centrality": partial(nx.betweenness_centrality, weight="Weigth"),
        "eigenvector_centrality": partial(nx.eigenvector_centrality, weight="Weigth"),
    }

    G = cm.df_to_graph(df)
    measure_df = cm.calc_measures(G, measure_dict)

    write_measures(measure_df, filepath)
    visualize.plot_graph(G, filepath)


def main(args):
    datapath = Path(args.data_path)
    if datapath.is_file():
        process_file(datapath)
    elif datapath.is_dir():
        process_dir(datapath)
    else:
        raise FileNotFoundError


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Does rudimentary network analysis on a file or folder. The files must be tsv files with the following columns: 'Source', 'Target', 'Weight'"
    )

    argparser.add_argument(
        "--data-path",
        required=True,
        type=str,
        help="Path to either the directory or a specific tsv file",
    )
    args = argparser.parse_args()
    main(args)
