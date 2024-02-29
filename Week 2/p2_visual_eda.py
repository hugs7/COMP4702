"""
Week 2 Prac
Part 2: Visualisation of Data and EDA
"""

import os
import p1_loading
from colorama import Fore, Style
import visualise_data

def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_folder, "data")

    pokemon_data_file = os.path.join(data_folder, "pokemonsrt.csv")
    seedlings_data_file = os.path.join(data_folder, "seedlings.csv")
    snakes_data_file = os.path.join(data_folder, "Snakes.xlsx")

    data_files = [snakes_data_file, pokemon_data_file, seedlings_data_file ]

    for data_file in data_files:
        data_processed = p1_loading.load_and_process_data(data_file, replace_null=True)
        
        # Visualise the data
        print(f"{Fore.LIGHTBLUE_EX}Visualising Data{Style.RESET_ALL}")
        visualise_data.plot_data_pairs(data_processed)

        print("-"*100)


if __name__ == "__main__":
    main()
