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


def accuracy_to_error(accuracy: float) -> float:
    """
    Converts an accuracy score to an error score.

    Parameters:
    - accuracy (float): The accuracy score to convert.

    Returns:
    - float: The error score.
    """

    return 1 - accuracy


def col_names_to_indices(all_col_names: List[str], col_names: List[str]) -> List[int]:
    """
    Converts a list of column names to their corresponding indices in a DataFrame.

    Parameters:
    - all_col_names (List[str]): All the column names in the DataFrame.
    - col_names (List[str]): The column names to search for.

    Returns:
    - List[int]: The indices of the column names in the DataFrame.
    """

    return [all_col_names.index(col) for col in col_names]
