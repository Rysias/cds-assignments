import math
import itertools
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from multiprocessing import Pool

import src.img_help as ih


def calc_color_hist(img: np.ndarray) -> np.ndarray:
    """ Finds the color histogram of an image """
    return cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])


def create_norm_hist(img: np.ndarray) -> np.ndarray:
    """ Creates a normalized histogram from an image """
    hist = calc_color_hist(img)
    return cv2.normalize(hist, hist).flatten()


def process_img(img_path: Path) -> np.ndarray:
    """Reads image and returns the normalized histogram"""
    img = ih.read_img(img_path)
    return create_norm_hist(img)


def create_hist_dict(img_path: Path) -> Dict[Path, np.ndarray]:
    """ Associates each image path with a normalized histogram """
    return {img_path: process_img(img_path)}


def compare_hists(source_hist: np.ndarray, candidate_hist: np.ndarray) -> float:
    """ Finds the chi-squared distance between two histograms """
    return cv2.compareHist(source_hist, candidate_hist, cv2.HISTCMP_CHISQR)


def create_dist_df(
    source_path: Path, hist_dict: Dict[Path, np.ndarray]
) -> pd.DataFrame:
    dist_df = pd.DataFrame(
        {"dist": 0}, index=(key for key in hist_dict if key != source_path)
    )
    source_hist = hist_dict[source_path]
    for cand_path, cand_hist in hist_dict.items():
        if cand_path == source_path:
            continue
        dist_df.loc[cand_path, "dist"] += compare_hists(source_hist, cand_hist)
    return dist_df


def filter_top_dists(dist_df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return dist_df.nsmallest(n=n, columns="dist")


def find_top_dists(
    target_img: Path, hist_dict: Dict[Path, np.ndarray], n=3
) -> pd.DataFrame:
    dist_df = create_dist_df(target_img, hist_dict)
    return filter_top_dists(dist_df, n=n)


def create_output_dir() -> Path:
    """Creates a directory for the output and returns the path """
    output_dir = Path("output")
    try:
        output_dir.mkdir()
    except FileExistsError:
        pass
    return output_dir


def list_to_dict(L: List[Dict]) -> dict:
    return {k: v for d in L for k, v in d.items()}


def create_master_hists(
    all_img_paths: List[Path], n_cores=10
) -> Dict[Path, np.ndarray]:
    """ Parallelized function for calculating histograms for all images in list"""
    with Pool(n_cores) as p:
        master_list = p.map(create_hist_dict, all_img_paths)
    return list_to_dict(master_list)


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
        "dist": compare_hists(pair1[1], pair2[1]),
    }


def find_all_similarities(
    hist_dict: Dict[Path, np.ndarray], n_cores: int
) -> pd.DataFrame:
    """Creates a full similarity dict based on all combinations of images """
    all_hist_pairs = itertools.combinations(hist_dict.items(), 2)
    with Pool(n_cores) as p:
        master_list = p.map(find_similarity, all_hist_pairs)
    return pd.DataFrame(master_list)


def combine_reversed_df(df: pd.DataFrame) -> pd.DataFrame:
    """A method for getting the permutations from a dataframe with combinations"""
    reversed_df = df.rename({"path1": "path2", "path2": "path1"}, axis=1)
    return pd.concat((df, reversed_df))


def find_smallest_cands(df, n=3, col="dist", target="path2"):
    """Finds the n smallest values from the specified column"""
    return df.nsmallest(n, col)[[target]].assign(rank=["1st", "2nd", "3rd"])


def create_closest_df(dist_df: pd.DataFrame) -> pd.DataFrame:
    """Finds the 3 closest images for each image in the dataframe """
    full_df = combine_reversed_df(dist_df)
    grouped_df = full_df.groupby("path1").apply(find_smallest_cands).reset_index()
    return grouped_df.pivot(index="path1", columns="rank", values="path2")
