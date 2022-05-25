from pathlib import Path
import json


def read_json(filename: Path) -> dict:
    """
    Reads a json file.
    """
    with open(filename, "r") as f:
        return json.load(f)
