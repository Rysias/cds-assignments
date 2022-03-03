import argparse
import itertools
import find_similar_imgs as fsi
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from multiprocessing import cpu_count, Pool

def create_key(path1: Path, path2: Path) -> str: 
    return "-".join(sorted([path1.name, path2.name]))

def find_similarity(pair_tuble: Tuple[Tuple[Path, np.ndarray], Tuple[Path, np.ndarray]]): 
    pair1 = pair_tuble[0]
    pair2 = pair_tuble[1]
    return {"key": create_key(pair1[0], pair2[0]), "dist": fsi.compare_hists(pair1[1], pair2[1])}

def find_all_similarities(hist_dict: List[Path], n_cores: int) -> List[Dict]: 
    all_hist_pairs = itertools.combinations(hist_dict.items(), 2)
    with Pool(n_cores) as p:
        master_list = p.map(find_similarity, all_hist_pairs)
    return master_list

def main(args):
    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIR = fsi.create_output_dir()
    ncores = args.ncores if args.ncores is not None else cpu_count() - 1

    all_img_paths = list(DATA_DIR.glob("*.jpg"))

    # Heavy lifting y'all!
    master_dict = fsi.create_master_hists(all_img_paths, n_cores=ncores)
    all_dists = find_all_similarities(master_dict, ncores)
    print(all_dists[:5])
    
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Finds most similar images for all images in a directory. Output a csv with the columns 'source', '1st', '2nd' and '3rd'"
    )
    argparser.add_argument(
        "--data-dir",
        default="../../../CDS-VIS/flowers",
        help="Path to directory for finding images and similar images (optional)",
    )
    argparser.add_argument(
        "--ncores",
        type=int, 
        help="How many cores to use for multiprocessing (defaults to 1 less than total cpu)",
    )

    args = argparser.parse_args()
    main(args)
