"""
Week 2 - Prac
Part 1: Data Loading and Pre-processing
"""

import pandas as pd
import os

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

    print("Printing Data")
    if head_only:
        print(data.head())
    else:
        for column in data.columns:
            print(f"Values: \n{data[column]}")
            print("\n")


def remove_null_values(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna()


def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))

    pokemon_data_file = os.path.join(current_folder, "pokemonsrt.csv")
    seedlings_data_file = os.path.join(current_folder, "seedlings.csv")
    snakes_data_file = os.path.join(current_folder, "Snakes.xlsx")

    data_files = [pokemon_data_file, seedlings_data_file, snakes_data_file]

    for data_file in data_files:
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"File {data_file} not found")
        
        data = load_data(data_file)
        print(f"Data loaded from {data_file}")
        display_data(data, print_metadata=False, head_only = True)



if __name__ == "__main__":
    main()
