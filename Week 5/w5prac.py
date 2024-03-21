"""
Main file for the Week 5 Practical class
"""

from typing import Union
from pandas import DataFrame
import os

import load_data


def process_classification_data(
    data_folder,
) -> Union[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, list[str]]:
    # Load the data
    file_path = os.path.join(data_folder, "w3classif.csv")

    data = load_data.load_data(file_path)

    data = load_data.tag_data(data, ["X1", "X2", "Y"])

    return data


def q1(data_folder: str):
    classif_data = process_classification_data(data_folder)

    train_test_sets = {}

    # Randomise data 10 times
    for i in range(10):
        # Randomise the data
        data_randomised = load_data.shuffle_data(classif_data)

        feature_names = ["X1", "X2"]
        # Split the data into training and testing data
        X = data_randomised.loc[:, feature_names]
        y = data_randomised.loc[:, "Y"]
        ratio = 0.3

        X_train, y_train, X_test, y_test = load_data.test_train_split(X, y, ratio=ratio)

        # Add the data to the dictionary
        train_test_sets[i] = (X_train, y_train, X_test, y_test, feature_names)

    return train_test_sets


def main():
    current_folder = os.path.dirname(__file__)
    data_folder = os.path.join(current_folder, "data")

    # Question 1
    q1(data_folder)


if __name__ == "__main__":
    main()
