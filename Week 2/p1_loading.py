"""
Week 2 - Prac
Part 1: Data Loading and Pre-processing
"""

import os
from colorama import Fore, Style
import pandas as pd

import load_data
import display_data
import replace_values
import encode_data

def check_file_exists(file_path: str) -> bool:
    file_exists = os.path.exists(file_path)
    if not file_exists:
        raise FileNotFoundError(f"File {file_path} not found")

def load_and_process_data(data_file: str, replace_null: bool = False) -> pd.DataFrame:
    check_file_exists(data_file)

    # Load Data
    print(f"{Fore.LIGHTBLUE_EX}Loading Data{Style.RESET_ALL}")
    data = load_data.load_data(data_file)
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


def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_folder, "data")

    pokemon_data_file = os.path.join(data_folder, "pokemonsrt.csv")
    seedlings_data_file = os.path.join(data_folder, "seedlings.csv")
    snakes_data_file = os.path.join(data_folder, "Snakes.xlsx")

    data_files = [pokemon_data_file, seedlings_data_file, snakes_data_file]

    for data_file in data_files:
        data_processed = load_and_process_data(data_file, replace_null=True)
        
        display_data.display_data(data_processed, print_metadata=False, head_only = True)

        print("-"*100)

if __name__ == "__main__":
    main()
