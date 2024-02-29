"""
Week 2 - Prac
Part 1: Data Loading and Pre-processing
"""

import pandas as pd
import os
from colorama import Fore, Style
import numpy as np

def load_data(file_path: str)-> pd.DataFrame:
    df = None

    # If Excel file
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)

    # If CSV file
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)


    return df


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

    return data


def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))

    pokemon_data_file = os.path.join(current_folder, "pokemonsrt.csv")
    seedlings_data_file = os.path.join(current_folder, "seedlings.csv")
    snakes_data_file = os.path.join(current_folder, "Snakes.xlsx")

    data_files = [pokemon_data_file, seedlings_data_file, snakes_data_file]

    for data_file in data_files:
        print(f"{Fore.LIGHTGREEN_EX}Data File: {Fore.MAGENTA}{data_file}{Style.RESET_ALL}")
        # Check if file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"{Fore.RED}File {data_file} not found{Fore.WHITE}")
        
        # Load Data
        print(f"{Fore.LIGHTBLUE_EX}Loading Data{Style.RESET_ALL}")
        data = load_data(data_file)
        print(f"Data loaded from {data_file}")

        # Display Data
        # display_data(data, print_metadata=False, head_only = True)

        # Remove null values
        print(f"{Fore.LIGHTBLUE_EX}Removing Null Values{Style.RESET_ALL}")
        data_null_removed = remove_null_values(data)

        # Display Data
        # display_data(data_null_removed, print_metadata=False, head_only = True)

        # Replace null values with mean instead
        print(f"{Fore.LIGHTBLUE_EX}Replacing Null Values with Mean{Style.RESET_ALL}")
        data_null_replaces = replace_null_values_with_mean(data)

        # Display Data
        display_data(data_null_replaces, print_metadata=False, head_only = True)


        print("-"*100)

if __name__ == "__main__":
    main()
