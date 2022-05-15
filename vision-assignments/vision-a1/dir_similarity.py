import argparse
import logging
from pathlib import Path
from multiprocessing import cpu_count

import src.calculate_dists as cd
import src.format_output as fo


def main(args):
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO,
    )

    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIR = fo.create_output_dir()
    ncores = args.ncores if args.ncores is not None else cpu_count() - 1

    all_img_paths = list(DATA_DIR.glob("*.jpg"))

    # Heavy lifting y'all!
    logging.info("Calculating all histograms...")
    master_dict = cd.create_master_hists(all_img_paths, n_cores=ncores)
    logging.info("Calculating all distances...")
    all_dists = cd.find_all_similarities(master_dict, ncores)

    # Find closest for all
    logging.info("Finding closest to everyone...")
    closest_df = cd.create_closest_df(all_dists)
    logging.info("Writing output...")
    closest_df.to_csv(OUTPUT_DIR / f"{DATA_DIR.parent.name}_dists.csv")
    logging.info("All done!")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Finds most similar images for all images in a directory. Output a csv with the columns 'source', '1st', '2nd' and '3rd'"
    )
    argparser.add_argument(
        "--data-dir",
        default="input/jpg/",
        help="Path to directory for finding images and similar images (optional; see default)",
    )
    argparser.add_argument(
        "--ncores",
        type=int,
        help="How many cores to use for multiprocessing (optional; defaults to 1 less than total cpu)",
    )

    args = argparser.parse_args()
    main(args)
