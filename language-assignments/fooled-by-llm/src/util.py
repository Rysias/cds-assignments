"""
Miscellaneous functions for reading and writing files.
"""

from pathlib import Path
import json


def create_dir(dir: str) -> Path:
    """
    Creates a directory if it doesn't exist.
    """
    path = Path(dir)
    if not path.exists():
        path.mkdir(parents=True)
    return path


def read_json(filename: Path) -> dict:
    """
    Reads a json file.
    """
    with open(filename, "r") as f:
        return json.load(f)
