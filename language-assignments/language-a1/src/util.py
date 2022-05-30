"""
Miscellaneous functions for interacting with files.
"""

import pandas as pd
from pathlib import Path
from typing import List


def read_txt(file_path: Path) -> List[str]:
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return f.read().splitlines()


def clean_file(file_path: Path) -> str:
    raw_text = read_txt(file_path)
    return " ".join(raw_text).lower()


def write_output(collocate_df: pd.DataFrame, file_name: Path, search_term):
    output_name = Path("output") / f"{file_name.stem}_{search_term}.csv"
    collocate_df.to_csv(output_name, index=False)
