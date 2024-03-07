"""
Helper file to load data using pandas
"""

import pandas as pd
from pandas import DataFrame


def load_data(file_path: str) -> pd.DataFrame:
    df = None

    # If Excel file
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)

    # If CSV file
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

    return df


def tag_data(data: DataFrame) -> DataFrame:
    """
    Tags the data as first column X, second column Y and third column class
    """

    data.columns = ["X1", "X2", "Y"]

    return data
