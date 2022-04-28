import math
import itertools
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from multiprocessing import Pool, cpu_count

import calculate_dists as cd
import img_help as ih


def add_text(img: np.ndarray, text: str) -> np.ndarray:
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


def format_source_img(source_path: Path) -> np.ndarray:
    """ Makes source image into a square and add the word 'SOURCE' to the middle """
    return add_text(resize_square(ih.read_img(source_path)), "SOURCE")


def format_dist_list(
    source_path: Path, small_dist_df: pd.DataFrame
) -> List[np.ndarray]:
    """ Formats all images for the final square image """
    dist_img_list = [format_source_img(source_path)]
    for filename, row in small_dist_df.iterrows():
        img = resize_square(ih.read_img(filename))
        dist_img_list.append(add_text(img, format_dist(row["dist"])))
    return dist_img_list


def arrange_square(img_list: List[np.ndarray], img_dim=300) -> np.ndarray:
    """Adapted from https://stackoverflow.com/a/52283965
    Arranges images into a square
    """
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


def create_dist_output_dict(target_file: Path, dist_df: pd.DataFrame) -> pd.DataFrame:
    dist_dict = {
        "source": [target_file.name],
        "1st": [dist_df.index[0].name],
        "2nd": [dist_df.index[1].name],
        "3rd": [dist_df.index[2].name],
    }
    return pd.DataFrame.from_dict(dist_dict)


def create_dist_square(source_path, small_dist_df: pd.DataFrame) -> np.ndarray:
    """ Creates the output image with source image and three closest ones with info arranged in a square"""
    formatted_imgs = format_dist_list(source_path, small_dist_df)
    return arrange_square(formatted_imgs)

