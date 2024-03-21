"""
Main file for the Week 5 Practical class
"""

from typing import Dict, Union
from pandas import DataFrame
import os

import load_data

from colorama import Fore, Style

current_folder = os.path.dirname(__file__)
parent_folder = os.path.dirname(current_folder)
import knn


def print_question_header(question_num: int) -> None:
    print(f"{Fore.LIGHTYELLOW_EX}Question {question_num}:{Style.RESET_ALL}")


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
    print(f"{Fore.LIGHTMAGENTA_EX}Training accuracy: {Style.RESET_ALL}", train_accuracy)
    print(f"{Fore.LIGHTMAGENTA_EX}Testing accuracy: {Style.RESET_ALL}", test_accuracy)

    if show_mcr:
        print("")
        print(
            f"{Fore.LIGHTMAGENTA_EX}Training MCR: {Style.RESET_ALL}", 1 - train_accuracy
        )
        print(
            f"{Fore.LIGHTMAGENTA_EX}Testing MCR: {Style.RESET_ALL}", 1 - test_accuracy
        )


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


def q1(data_folder: str, split_ratio: float = 0.3, print_header: bool = True):
    """
    Repeat Q2 from Prac W3 10 times, saving the 10 resulting training and test sets.
    """
    if print_header:
        print_question_header(1)

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
        train_accuracy, test_accuracy = knn_classify(
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


def sample_mean(data: list[float]) -> float:
    """
    Calculate the sample mean of a list of numbers
    """

    return sum(data) / len(data)


def sample_standard_deviation(data: list[float]) -> float:
    """
    Calculate the sample standard deviation of a list of numbers
    """

    n = len(data)

    mean = sum(data) / n

    variance = sum((x - mean) ** 2 for x in data) / (n - 1)

    return variance**0.5


def q4(
    train_accuracies_q2, test_accuracies_q2, train_accuracies_q3, test_accuracies_q3
):
    """
    Calculate the sample standard deviation of your training and test
    set error values over the 10 trials from Q2 and Q3. What do you observe?
    """

    print_question_header(4)

    # Question 2

    q2_train_accuracies_sample_mean = sample_mean(train_accuracies_q2)
    q2_train_accuracies_sample_std = sample_standard_deviation(train_accuracies_q2)

    # Question 3

    q3_test_accuracies_sample_mean = sample_mean(test_accuracies_q3)
    q3_test_accuracies_sample_std = sample_standard_deviation(test_accuracies_q3)

    print(
        f"{Fore.LIGHTCYAN_EX}Sample mean of training accuracies from Q2: {Style.RESET_ALL}",
        q2_train_accuracies_sample_mean,
    )
    print(
        f"{Fore.LIGHTCYAN_EX}Sample standard deviation of training accuracies from Q2: {Style.RESET_ALL}",
        q2_train_accuracies_sample_std,
    )

    print(
        f"{Fore.LIGHTCYAN_EX}Sample mean of test accuracies from Q3: {Style.RESET_ALL}",
        q3_test_accuracies_sample_mean,
    )
    print(
        f"{Fore.LIGHTCYAN_EX}Sample standard deviation of test accuracies from Q3: {Style.RESET_ALL}",
        q3_test_accuracies_sample_std,
    )


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
