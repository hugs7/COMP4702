"""
Helper to read data from Excel format
"""

from pandas import DataFrame
import pandas as pd
import os
from colorama import Fore, Style


def check_file_exists(file_path: str) -> bool:
    """
    Check if the file exists

    Returns True if the file exists, False otherwise
    """

    return os.path.isfile(file_path)


def read_excel_file(file_path: str, header_row: int = 0) -> DataFrame:
    if not check_file_exists(file_path):
        raise FileNotFoundError(
            f"{Fore.RED}File {file_path} not found{Style.RESET_ALL}"
        )

    # If Excel file
    if file_path.endswith(".xlsx"):
        return pd.read_excel(file_path, header=[header_row])

    # If CSV file
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path, header=[header_row])

    file_format = file_path.split(".")[-1]

    raise Exception(
        f"{Fore.RED}File format not supported: {file_format}{Style.RESET_ALL}"
    )
