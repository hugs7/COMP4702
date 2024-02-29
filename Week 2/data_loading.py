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


def display_data(data: pd.DataFrame, print_metadata: bool = False):
    if print_metadata:
        print("Metadata")
        for column in data.columns:
            print(f"Column: {column}")
            print(f"Data type: {data[column].dtype}")
            print(f"Number of null values: {data[column].isnull().sum()}")
            print(f"Number of unique values: {data[column].nunique()}")

    print("Printing Data")
    for column in data.columns:
        print(f"Values: \n{data[column]}")
        print("\n")



def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))

    pokemon_data_file = os.path.join(current_folder, "pokemonsrt.csv")
    seedlings_data_file = os.path.join(current_folder, "seedlings.csv")
    snakes_data_file = os.path.join(current_folder, "Snakes.xlsx")

    pokemon_data = load_data(pokemon_data_file)
    display_data(pokemon_data)
    # Data is partly text and mostly numerical floats

    seedlings_data = load_data(seedlings_data_file)
    display_data(seedlings_data)
    # Data is mostly numerical with some text

    snakes_data = load_data(snakes_data_file)
    display_data(snakes_data)
    # Data is a mix of numerical, categorical and text based.


if __name__ == "__main__":
    main()
