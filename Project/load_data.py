"""
Helper file to load data using pandas
"""

import pandas as pd
from pandas import DataFrame
import os


def load_data(file_path: str) -> pd.DataFrame:
    df = None

    # If Excel file
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)

    # If CSV file
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

    return df


def tag_data(data: DataFrame, cols: list[str]) -> DataFrame:
    """
    Tags the data as first column X, second column Y and third column class
    """

    data.columns = cols

    return data


def shuffle_data(data: DataFrame) -> DataFrame:
    """
    Shuffles the data randomly.

    Parameters:
    - data: A pandas DataFrame containing the data to be shuffled.

    Returns:
    - A new pandas DataFrame with the data shuffled randomly.
    """

    return data.sample(frac=1).reset_index(drop=True)


def file_exists(file_path: str) -> bool:
    """
    Check if a file exists.

    Parameters:
    - file_path (str): The file path.

    Returns:
    - bool: True if the file exists, False otherwise.
    """

    return os.path.exists(file_path)
