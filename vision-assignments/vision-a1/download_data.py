"""
Downloads the flower dataset and puts them in the input folder
"""
import urllib.request
import tarfile
import os
import logging
from pathlib import Path


def download_tar(url, path):
    # Download the dataset
    logging.info("Downloading the dataset...")
    urllib.request.urlretrieve(url, path)
    logging.info("Download complete!")
    # Extract the dataset
    logging.info("Extracting the dataset...")
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(path=INPUT_PATH)
    logging.info("Extraction complete!")


# basic logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

FLOWER_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"
INPUT_PATH = Path("input")

if __name__ == "__main__":
    # Download the dataset
    if not os.path.exists(INPUT_PATH):
        os.makedirs(INPUT_PATH)
    download_tar(FLOWER_URL, INPUT_PATH / "17flowers.tgz")
