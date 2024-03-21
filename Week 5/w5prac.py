"""
Main file for the Week 5 Practical class
"""

from typing import Dict, Union
from pandas import DataFrame
import os
from colorama import Fore, Style

import load_data
import stats
import knn_helper
from print_helper import (
    print_question_header,
    print_sample_mean,
    print_sample_standard_deviation,
)

current_folder = os.path.dirname(__file__)
parent_folder = os.path.dirname(current_folder)


def q1(data_folder: str, split_ratio: float = 0.3, print_header: bool = True):
    """
    Repeat Q2 from Prac W3 10 times, saving the 10 resulting training and test sets.
    """
    if print_header:
        print_question_header(1)

    classif_data = load_data.process_classification_data(data_folder)

    train_test_sets = {}

    # Randomise data 10 times
    for i in range(10):
        # Randomise the data
        data_randomised = load_data.shuffle_data(classif_data)

        feature_names = ["X1", "X2"]
        # Split the data into training and testing data
        X = data_randomised.loc[:, feature_names]
        y = data_randomised.loc[:, "Y"]

        X_train, y_train, X_test, y_test = load_data.test_train_split(
            X, y, ratio=split_ratio
        )

        # Add the data to the dictionary
        train_test_sets[i] = (X_train, y_train, X_test, y_test, feature_names)

    return train_test_sets


def q2(
    train_test_sets: Dict[
        int, tuple[DataFrame, DataFrame, DataFrame, DataFrame, list[str]]
    ],
    print_header: bool = True,
) -> tuple[list[float], list[float]]:
    """
    Calculate the training and test set errors over all of the datasets from Q1 and calculate the average
    training and test errors over the 10 trials. Are the averages lower or higher than the values you found
    in Prac W3 (or alternatively compare with the values for the first of your 10 runs)?
    """

    if print_header:
        print_question_header(2)

    train_accuracies = []
    test_accuracies = []

    for run_num, data in train_test_sets.items():
        print(f"{Fore.LIGHTGREEN_EX}Run {run_num + 1}:{Style.RESET_ALL}")
        X_train, y_train, X_test, y_test, feature_names = data

        # Train the model
        train_accuracy, test_accuracy = knn_helper.knn_classify(
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            plot=False,
        )

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print()

    return train_accuracies, test_accuracies


def q3(data_folder: str) -> tuple[list[float], list[float]]:
    """
    Repeat Q1 and Q2 but use a different split – try 50/50 or 90/10.
    Compare your average error values with those you found in Q2.
    """

    print_question_header(3)

    split_ratio = 0.05

    train_test_sets = q1(data_folder, split_ratio=split_ratio, print_header=False)

    train_accuracies, test_accuracies = q2(train_test_sets, False)

    return train_accuracies, test_accuracies


def q4(
    train_accuracies_q2, test_accuracies_q2, train_accuracies_q3, test_accuracies_q3
):
    """
    Calculate the sample standard deviation of your training and test
    set error values over the 10 trials from Q2 and Q3. What do you observe?
    """

    print_question_header(4)

    # Question 2

    q2_train_accuracies_sample_mean = stats.sample_mean(train_accuracies_q2)
    q2_train_accuracies_sample_std = stats.sample_standard_deviation(
        train_accuracies_q2
    )

    q2_test_accuracies_sample_mean = stats.sample_mean(test_accuracies_q2)
    q2_test_accuracies_sample_std = stats.sample_standard_deviation(test_accuracies_q2)

    # Question 3

    q3_train_accuracies_sample_mean = stats.sample_mean(train_accuracies_q3)
    q3_train_accuracies_sample_std = stats.sample_standard_deviation(
        train_accuracies_q3
    )

    q3_test_accuracies_sample_mean = stats.sample_mean(test_accuracies_q3)
    q3_test_accuracies_sample_std = stats.sample_standard_deviation(test_accuracies_q3)

    # Print
    print(f"{Fore.LIGHTGREEN_EX}Question 2:{Style.RESET_ALL}")
    print("Train Accuracies:")
    print_sample_mean(2, q2_train_accuracies_sample_mean)
    print_sample_standard_deviation(2, q2_train_accuracies_sample_std)

    print("Test Accuracies:")
    print_sample_mean(2, q2_test_accuracies_sample_mean)
    print_sample_standard_deviation(2, q2_test_accuracies_sample_std)

    print(f"{Fore.LIGHTGREEN_EX}Question 3:{Style.RESET_ALL}")
    print("Train Accuracies:")
    print_sample_mean(3, q3_train_accuracies_sample_mean)
    print_sample_standard_deviation(3, q3_train_accuracies_sample_std)

    print("Test Accuracies:")
    print_sample_mean(3, q3_test_accuracies_sample_mean)
    print_sample_standard_deviation(3, q3_test_accuracies_sample_std)


def q5():
    """
    Perform 10-fold cross validation using your model and the (original) dataset (use existing
    Matlab or python functions to do this). What are the mean and standard devations of the
    cross-validation error?
    """


def main():
    data_folder = os.path.join(current_folder, "data")

    # Question 1
    train_test_sets = q1(data_folder)

    # Question 2
    train_accuracies_q2, test_accuracies_q2 = q2(train_test_sets)

    # Question 3
    train_accuracies_q3, test_accuracies_q3 = q3(data_folder)

    # Question 4
    q4(train_accuracies_q2, test_accuracies_q2, train_accuracies_q3, test_accuracies_q3)


if __name__ == "__main__":
    main()
