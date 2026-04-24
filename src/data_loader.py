import sys
from os import PathLike
from pathlib import Path

import pandas as pd


def load_data(path: str | PathLike[str]) -> pd.DataFrame:
    if not Path(path).exists():
        print(f"Error path not exists : {path}")
        sys.exit(1)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("Error could not found file.")
        sys.exit(1)
    else:
        print("OK loaded file :")

    return df
