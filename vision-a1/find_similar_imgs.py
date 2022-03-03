import argparse
import math
import itertools
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from multiprocessing import Pool, cpu_count


def calc_color_hist(img, key=None):
    return cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])


def create_norm_hist(img: np.ndarray, key=None) -> np.ndarray:
    hist = calc_color_hist(img, key=key)
    return cv2.normalize(hist, hist).flatten()


def read_img(path: Path) -> np.ndarray:
    return cv2.imread(str(path))


def process_img(img_path: Path):
    img = read_img(img_path)
    return create_norm_hist(img)


def create_hist_dict(img_path: Path) -> Dict[Path, np.ndarray]:
    return {img_path: process_img(img_path)}


def compare_hists(source_hist, candidate_hist):
    return cv2.compareHist(source_hist, candidate_hist, cv2.HISTCMP_CHISQR)


def create_dist_df(
    source_path: Path, hist_dict: Dict[Path, np.ndarray]
) -> pd.DataFrame:
    dist_df = pd.DataFrame({"dist": 0}, index=(key for key in hist_dict.keys() if key != source_path))
    source_hist = hist_dict[source_path]
    for cand_path, cand_hist in hist_dict.items():
        if cand_path == source_path:
            continue
        dist_df.loc[cand_path, "dist"] += compare_hists(source_hist, cand_hist)
    return dist_df


def create_dist_output_dict(target_file: Path, dist_df: pd.DataFrame) -> pd.DataFrame:
    dist_dict = {
        "source": [target_file.name],
        "1st": [dist_df.index[0].name],
        "2nd": [dist_df.index[1].name],
        "3rd": [dist_df.index[2].name],
    }
    return pd.DataFrame.from_dict(dist_dict)


def filter_top_dists(dist_df: pd.DataFrame, n=3) -> pd.DataFrame:
    return dist_df.nsmallest(n=n, columns="dist")


def find_top_dists(
    target_img: Path, hist_dict: Dict[Path, np.ndarray], n=3
) -> pd.DataFrame:
    dist_df = create_dist_df(target_img, hist_dict)
    return filter_top_dists(dist_df, n=n)


def add_text(img, text):
    new_img = img.copy()
    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # get coords based on boundary
    textX = (img.shape[1] - textsize[0]) // 2
    textY = (img.shape[0] + textsize[1]) // 2

    # add text centered on image
    cv2.putText(new_img, text, (textX, textY), font, 1, (255, 255, 255), 2)

    return new_img


def format_dist(dist: float) -> str:
    return f"dist:{dist: .2f}"


def resize_square(img: np.ndarray, size=300) -> np.ndarray:
    return cv2.resize(img, dsize=(size, size))


def arrange_square(img_list: List[np.ndarray], img_dim=300) -> np.ndarray:
    """Adapted from https://stackoverflow.com/a/52283965"""
    if len(img_list) not in [i ** 2 for i in range(10)]:
        raise ValueError("List must have square number of elements")

    canvas_shape = math.isqrt(len(img_list))
    imgmatrix = np.zeros((canvas_shape * img_dim, canvas_shape * img_dim, 3), np.uint8)
    # Prepare an iterable with the right dimensions
    positions = itertools.product(range(canvas_shape), range(canvas_shape))

    for (y_i, x_i), img in zip(positions, img_list):
        x = x_i * img_dim
        y = y_i * img_dim
        imgmatrix[y : y + img_dim, x : x + img_dim, :] = img
    return imgmatrix


def format_source_img(source_path: Path) -> np.ndarray:
    return add_text(resize_square(read_img(source_path)), "SOURCE")


def format_dist_list(
    source_path: Path, small_dist_df: pd.DataFrame
) -> List[np.ndarray]:
    dist_img_list = [format_source_img(source_path)]
    for filename, row in small_dist_df.iterrows():
        img = resize_square(read_img(filename))
        dist_img_list.append(add_text(img, format_dist(row["dist"])))
    return dist_img_list


def create_dist_square(source_path, small_dist_df: pd.DataFrame) -> np.ndarray:
    formatted_imgs = format_dist_list(source_path, small_dist_df)
    return arrange_square(formatted_imgs)


def create_output_dir() -> Path:
    output_dir = Path("output")
    try:
        output_dir.mkdir()
    except FileExistsError:
        pass
    return output_dir


def write_output_img(output_img: np.ndarray, source_path: Path, OUTPUT_DIR) -> None:
    output_name = str(OUTPUT_DIR / f"{source_path.stem}_closest3.png")
    cv2.imwrite(output_name, output_img)


def list_to_dict(L: List[Dict]) -> dict:
    return {k: v for d in L for k, v in d.items()}

def create_master_hists(all_img_paths: List[Path], n_cores=10) -> Dict[Path, np.ndarray]: 
    with Pool(n_cores) as p:
        master_list = p.map(create_hist_dict, all_img_paths)
    return list_to_dict(master_list)


def main(args):
    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIR = create_output_dir()
    target_img = DATA_DIR / args.img_name
    ncores = args.ncores if args.ncores is not None else cpu_count() - 1

    all_img_paths = list(DATA_DIR.glob("*.jpg"))

    # Heavy lifting y'all!
    master_dict = create_master_hists(all_img_paths)
    
    dist_df = find_top_dists(target_img, master_dict)

    # format output
    output_df = create_dist_output_dict(target_img, dist_df)
    output_img = create_dist_square(target_img, dist_df)

    # writing output
    output_df.to_csv(OUTPUT_DIR / f"{target_img.stem}_closest3.csv")
    write_output_img(output_img, target_img, OUTPUT_DIR)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Given a filename, finds the top three similar images in the same directory. Outputs an image with th e source + top three similar, as well as a csv-file with the file names."
    )
    argparser.add_argument(
        "--img-name",
        required=True,
        help="file name of the specified image",
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