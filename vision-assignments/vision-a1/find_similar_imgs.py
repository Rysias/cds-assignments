import argparse
from pathlib import Path
from multiprocessing import cpu_count

import src.calculate_dists as cd
import src.format_output as fo


def main(args: argparse.Namespace) -> None:
    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIR = fo.create_output_dir()
    target_img = DATA_DIR / args.img_name
    ncores = args.ncores if args.ncores is not None else cpu_count() - 1

    all_img_paths = list(DATA_DIR.glob("*.jpg"))

    # Heavy lifting y'all!
    master_dict = cd.create_master_hists(all_img_paths, n_cores=ncores)
    dist_df = cd.find_top_dists(target_img, master_dict)

    # format output
    output_df = fo.create_dist_output_dict(target_img, dist_df)
    output_img = fo.create_dist_square(target_img, dist_df)

    # writing output
    output_df.to_csv(OUTPUT_DIR / f"{target_img.stem}_closest3.csv")
    fo.write_output_img(output_img, target_img, OUTPUT_DIR)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Given a filename, finds the top three similar images in the same directory. Outputs an image with th e source + top three similar, as well as a csv-file with the file names."
    )
    argparser.add_argument(
        "--img-name", required=True, help="file name of the specified image",
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
