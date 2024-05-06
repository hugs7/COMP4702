"""
Helper function to encode non-numeric data as numeric
"""

import pandas as pd
from logger import *


def encode_non_numeric_data(data: pd.DataFrame) -> pd.DataFrame:
    # Create a copy of the data
    data = data.copy()

    for column in data.columns:
        sample_data_point = data[column].iloc[0]
        log_trace(f"Column: {column}, Sample Data Point: {sample_data_point}, Data Type: {data[column].dtype}")
        if data[column].dtype == str or data[column].dtype == object:
            # Replace with mode
            data[column] = data[column].astype("category")
            data[column] = data[column].cat.codes

    return data
