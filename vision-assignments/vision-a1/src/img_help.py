from pathlib import Path
import numpy as np
import cv2


def read_img(path: Path) -> np.ndarray:
    return cv2.imread(str(path))
