"""
Main file for the Week 5 Practical class
"""

from typing import Dict, Union
from pandas import DataFrame
import os

import load_data

from colorama import Fore

current_folder = os.path.dirname(__file__)
parent_folder = os.path.dirname(current_folder)
import knn


def process_classification_data(
    data_folder,
) -> Union[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, list[str]]:
    # Load the data
    file_path = os.path.join(data_folder, "w3classif.csv")

    data = load_data.load_data(file_path)

    data = load_data.tag_data(data, ["X1", "X2", "Y"])

    return data


def print_classify_results(
    train_accuracy: float, test_accuracy: float, show_mcr: bool
) -> None:
    print(f"{Fore.LIGHTMAGENTA_EX}Training accuracy: {Fore.WHITE}", train_accuracy)
    print(f"{Fore.LIGHTMAGENTA_EX}Testing accuracy: {Fore.WHITE}", test_accuracy)

    if show_mcr:
        print("")
        print(f"{Fore.LIGHTMAGENTA_EX}Training MCR: {Fore.WHITE}", 1 - train_accuracy)
        print(f"{Fore.LIGHTMAGENTA_EX}Testing MCR: {Fore.WHITE}", 1 - test_accuracy)


def knn_classify(
    X_train, y_train, X_test, y_test, feature_names, plot: bool = False
) -> tuple[float, float]:
    """
    Classification task using knn
    """

    # Apply the knn classifier
    knn_classifier = knn.KNNClassify(
        X_train, y_train, X_test, y_test, feature_names, k=3
    )

    test_preds, train_accuracy, test_accuracy = knn_classifier.classify()

    print_classify_results(train_accuracy, test_accuracy, False)

    if plot:
        # Plot decision regions
        knn_plot_title = f"k-NN decision regions (k = {knn_classifier.get_k()})"
        knn_classifier.plot_decision_regions(
            test_preds, resolution=0.02, plot_title=knn_plot_title
        )

    return train_accuracy, test_accuracy


def q1(data_folder: str):
    """
    Repeat Q2 from Prac W3 10 times, saving the 10 resulting training and test sets.
    """

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


def q2(
    train_test_sets: Dict[
        int, tuple[DataFrame, DataFrame, DataFrame, DataFrame, list[str]]
    ]
):
    """
    Calculate the training and test set errors over all of the datasets from Q1 and calculate the average
    training and test errors over the 10 trials. Are the averages lower or higher than the values you found
    in Prac W3 (or alternatively compare with the values for the first of your 10 runs)?
    """

    for run_num, data in train_test_sets.items():
        print(f"{Fore.LIGHTGREEN_EX}Run {run_num + 1}:{Fore.WHITE}")
        X_train, y_train, X_test, y_test, feature_names = data

        # Train the model
        train_accuracy, test_accuracy = knn_classify(
            X_train, y_train, X_test, y_test, feature_names, plot=False
        )

        print()


def main():
    data_folder = os.path.join(current_folder, "data")

    # Question 1
    train_test_sets = q1(data_folder)

    # Question 2
    q2(train_test_sets)


if __name__ == "__main__":
    main()
