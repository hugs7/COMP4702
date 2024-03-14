"""
Helper function to replace values
"""

import pandas as pd
from colorama import Fore


def remove_null_values(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna()


def replace_null_values_with_mean(data: pd.DataFrame, supress_logs: bool = True) -> pd.DataFrame:
    # Create a copy of the data
    data = data.copy()

    for column in data.columns:
        if data[column].dtype in [int, float]:
            column_mean = data[column].mean()

            for row in range(len(data[column])):
                data_point = data.loc[row, column]
                if pd.isna(data_point):
                    if not supress_logs:
                        print(f"{Fore.LIGHTRED_EX}Replacing null value at column {column} and row {row} with mean value {column_mean}{Fore.WHITE}")
                    data.loc[row, column] = column_mean
        elif data[column].dtype == str:
            # Replace with mode
            column_mode = data[column].mode()[0]
            for row in range(len(data[column])):
                data_point = data.loc[row, column]
                if pd.isna(data_point):
                    if not supress_logs:
                        print(f"{Fore.LIGHTRED_EX}Replacing null value at column {column} and row {row} with mode value {column_mode}{Fore.WHITE}")
                    data.loc[row, column] = column_mode

    return data
