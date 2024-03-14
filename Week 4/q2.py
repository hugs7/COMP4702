"""
Defined the function for question 2
"""

import os

from load_data import load_data


def q2(data_folder: str):
    # Part A - Import the data

    file_path = os.path.join(data_folder, "pokemonregr.csv")

    data = load_data(file_path)

    print(data.head())

    print(data.info())
