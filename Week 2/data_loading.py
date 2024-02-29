"""
Week 2 - Prac
Part 1: Data Loading and Pre-processing
"""

import pandas as pd
import os
from colorama import Fore, Style
import numpy as np

import load_data
import display_data
import replace_values







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
        data = load_data.load_data(data_file)
        print(f"Data loaded from {data_file}")

        # Display Data
        # display_data(data, print_metadata=False, head_only = True)

        # Remove null values
        print(f"{Fore.LIGHTBLUE_EX}Removing Null Values{Style.RESET_ALL}")
        data_null_removed = replace_values.remove_null_values(data)

        # Display Data
        # display_data(data_null_removed, print_metadata=False, head_only = True)

        # Replace null values with mean instead
        print(f"{Fore.LIGHTBLUE_EX}Replacing Null Values with Mean{Style.RESET_ALL}")
        data_null_replaces = replace_values.replace_null_values_with_mean(data)

        # Display Data
        display_data(data_null_replaces, print_metadata=False, head_only = True)


        print("-"*100)

if __name__ == "__main__":
    main()
