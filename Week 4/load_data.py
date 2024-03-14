"""
Helper file to load data using pandas
"""

import pandas as pd

import os
from colorama import Fore, Style
import replace_values
import encode_data


def load_data(file_path: str) -> pd.DataFrame:
    df = None

    # If Excel file
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)

    # If CSV file
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

    return df


def check_file_exists(file_path: str) -> bool:
    file_exists = os.path.exists(file_path)
    if not file_exists:
        raise FileNotFoundError(f"File {file_path} not found")


def load_and_process_data(data_file: str, replace_null: bool = False) -> pd.DataFrame:
    check_file_exists(data_file)

    # Load Data
    print(f"{Fore.LIGHTBLUE_EX}Loading Data{Style.RESET_ALL}")
    data = load_data(data_file)
    print(f"Data loaded from {data_file}")

    if replace_null:
        # Replace null values with mean instead
        print(f"{Fore.LIGHTBLUE_EX}Replacing Null Values with Mean{Style.RESET_ALL}")
        data = replace_values.replace_null_values_with_mean(data)
    else:
        # Remove null values
        print(f"{Fore.LIGHTBLUE_EX}Removing Null Values{Style.RESET_ALL}")
        data = replace_values.remove_null_values(data)

    # Encode non-numeric data
    print(f"{Fore.LIGHTBLUE_EX}Encoding Non-Numeric Data{Style.RESET_ALL}")
    data_encoded = encode_data.encode_non_numeric_data(data)

    return data_encoded
