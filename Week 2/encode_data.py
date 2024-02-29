"""
Helper function to encode non-numeric data as numeric
"""


import pandas as pd


def encode_non_numeric_data(data: pd.DataFrame) -> pd.DataFrame:
    # Create a copy of the data
    data = data.copy()

    for column in data.columns:
        if data[column].dtype == str:
            # Replace with mode
            data[column] = data[column].astype('category')
            data[column] = data[column].cat.codes

    return data