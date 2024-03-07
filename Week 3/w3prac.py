"""
Week 3 Practical
Hugo Burton
07/03/2024
"""

import pandas as pd
import load_data
import os


def main():

    # Load the data
    current_folder = os.path.dirname(__file__)
    data_folder = "data"
    file_path = os.path.join(current_folder, data_folder, "w3classif.csv")

    data = load_data.load_data(file_path)

    print(data)


if __name__ == "__main__":
    main()
