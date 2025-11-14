"""
Class defines CSVLoader, which loads a given CSV at fpath into a dataframe.
"""
# == Third party imports ==
import pandas as pd
from pathlib import Path

class CSVLoader:
    def __init__(self, fpath: str):
        self.filepath = Path(fpath)

    def load(self) -> pd.DataFrame:
        """
        Method validates the given filepath and loads a stored CSV at that
        location as a dataframe.
        :return: Dataframe object of loaded CSV.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {self.filepath}")
        if not self.filepath.suffix.lower() == ".csv":
            raise ValueError(f"Invalid file type: {self.filepath.suffix}")
        return pd.read_csv(self.filepath)