"""
This file contains utility functions that are used throughout the project.
"""

from typing import Dict, List
import pandas as pd
import numpy as np
import logger


def key_from_value(dict: Dict, value: str) -> int:
    """
    Get the key from a dictionary by the value. IF multiple
    keys have the same value, the first key found will be returned.

    Args:
    - dict (Dict): The dictionary to search.
    - value (str): The value to search for.

    Returns:
    - int: The key of the value.
    """
    return [k for k, v in dict.items() if v == value][0]


def np_to_pd(df: np.ndarray, columns: List[str]) -> pd.DataFrame:
    """
    Converts a numpy array into a pandas DataFrame.

    Parameters:
    - df (ndarray): The numpy array to print.
    - columns (List[str]): The column names for the DataFrame.

    Returns:
    - DataFrame: The DataFrame representation of the numpy array.
    """

    df = pd.DataFrame(df, columns=columns)
    return df
