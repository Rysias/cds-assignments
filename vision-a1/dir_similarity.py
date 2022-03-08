import argparse
import itertools
import find_similar_imgs as fsi
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple, List
from multiprocessing import cpu_count, Pool


def find_similarity(
    pair_tuble: Tuple[Tuple[Path, np.ndarray], Tuple[Path, np.ndarray]]
) -> dict:
    """
    Finds the similarity between to images. The output is a Dictionary with 
    the ids (paths) and distance, suitable for a DataFrame
    """
    pair1 = pair_tuble[0]
    pair2 = pair_tuble[1]
    return {
        "path1": pair1[0],
        "path2": pair2[0],
        "dist": fsi.compare_hists(pair1[1], pair2[1]),
    }


def combine_reversed_df(df: pd.DataFrame) -> pd.DataFrame:
    """A method for getting the permutations from a dataframe with combinations"""
    reversed_df = df.rename({"path1": "path2", "path2": "path1"}, axis=1)
    return pd.concat((df, reversed_df))


def find_smallest_cands(df, n=3, col="dist", target="path2"):
    """Finds the n smallest values from the specified column"""
    return df.nsmallest(n, col)[[target]].assign(rank=["1st", "2nd", "3rd"])


def find_all_similarities(
    hist_dict: Dict[Path, np.ndarray], n_cores: int
) -> pd.DataFrame:
    """Creates a full similarity dict based on all combinations of images """
    all_hist_pairs = itertools.combinations(hist_dict.items(), 2)
    with Pool(n_cores) as p:
        master_list = p.map(find_similarity, all_hist_pairs)
    return pd.DataFrame(master_list)


def create_closest_df(dist_df: pd.DataFrame) -> pd.DataFrame:
    """Finds the 3 closest images for each image in the dataframe """
    full_df = combine_reversed_df(dist_df)
    grouped_df = full_df.groupby("path1").apply(find_smallest_cands).reset_index()
    return grouped_df.pivot(index="path1", columns="rank", values="path2")


def main(args):
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO,
    )

    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIR = fsi.create_output_dir()
    ncores = args.ncores if args.ncores is not None else cpu_count() - 1

    all_img_paths = list(DATA_DIR.glob("*.jpg"))

    # Heavy lifting y'all!
    logging.info("Calculating all histograms...")
    master_dict = fsi.create_master_hists(all_img_paths, n_cores=ncores)
    logging.info("Calculating all distances...")
    all_dists = find_all_similarities(master_dict, ncores)

    # Find closest for all
    logging.info("Finding closest to everyone...")
    closest_df = create_closest_df(all_dists)
    logging.info("Writing output...")
    closest_df.to_csv(OUTPUT_DIR / f"{DATA_DIR.parent.name}_dists.csv")
    logging.info("All done!")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Finds most similar images for all images in a directory. Output a csv with the columns 'source', '1st', '2nd' and '3rd'"
    )
    argparser.add_argument(
        "--data-dir",
        default="../../../CDS-VIS/flowers",
        help="Path to directory for finding images and similar images (optional; see default)",
    )
    argparser.add_argument(
        "--ncores",
        type=int,
        help="How many cores to use for multiprocessing (optional; defaults to 1 less than total cpu)",
    )

    args = argparser.parse_args()
    main(args)
