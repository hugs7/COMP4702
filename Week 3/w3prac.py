"""
Week 3 Practical
Hugo Burton
07/03/2024
"""

import os

import load_data
import scatterplot


def main():

    # Load the data
    current_folder = os.path.dirname(__file__)
    data_folder = "data"
    file_path = os.path.join(current_folder, data_folder, "w3classif.csv")

    data = load_data.load_data(file_path)

    data = load_data.tag_data(data)

    # Show the data as scatterplot
    scatterplot.scatterplot(data)


if __name__ == "__main__":
    main()
