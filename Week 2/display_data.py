"""
Helper file to display data
"""

import pandas as pd
from colorama import Fore


def display_data(data: pd.DataFrame, print_metadata: bool = False, head_only: bool = True):
    if print_metadata:
        print("Metadata")
        for column in data.columns:
            print(f"Column: {column}")
            print(f"Data type: {data[column].dtype}")
            print(f"Number of null values: {data[column].isnull().sum()}")
            print(f"Number of unique values: {data[column].nunique()}")

        print(data.info())

    print(f"{Fore.LIGHTBLUE_EX}Printing Data{Fore.WHITE}")
    if head_only:
        print(data.head())
    else:
        for column in data.columns:
            print(f"Values: \n{data[column]}")
            print("\n")

