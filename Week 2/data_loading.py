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


def display_data(data: pd.DataFrame):
    for column in data.columns:
        print(f"Column: {column}")
        print(f"Data type: {data[column].dtype}")
        print(f"Number of missing values: {data[column].isnull().sum()}")
        print(f"Number of unique values: {data[column].nunique()}")
        print(f"Unique values: {data[column].unique()}")
        print("\n")



def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))

    pokemon_data_file = os.path.join(current_folder, "pokemonsrt.csv")
    seedlings_data_file = os.path.join(current_folder, "seedlings.csv")
    snakes_data_file = os.path.join(current_folder, "Snakes.xlsx")

    pokemon_data = load_data(pokemon_data_file)
    display_data(pokemon_data)
    seedlings_data = load_data(seedlings_data_file)
    display_data(seedlings_data)
    snakes_data = load_data(snakes_data_file)
    display_data(snakes_data)


if __name__ == "__main__":
    main()
