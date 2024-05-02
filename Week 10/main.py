"""
Week 10 Practical - Ensemble Methods
Developer: Hugo Burton
Date: 02/05/2024
"""

import os
from typing import List, Union
from pandas import DataFrame
from colorama import Fore, Style
import matplotlib.pyplot as plt
import sys

import load_data
import decision_tree
import random_forest


def process_classification_data(data_folder, randomise: bool) -> Union[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, list[str]]:
    # Load the data
    file_path = os.path.join(data_folder, "w3classif.csv")
    data = load_data.load_data(file_path)
    data = load_data.tag_data(data, ["X1", "X2", "Y"])

    # Randomise the data
    if randomise:
        data = load_data.shuffle_data(data)

    feature_names = ["X1", "X2"]
    # Split the data into training and testing data
    X = data.loc[:, feature_names]
    y = data.loc[:, "Y"]
    ratio = 0.3

    X_train, y_train, X_test, y_test = load_data.test_train_split(X, y, ratio=ratio)

    return data, X_train, y_train, X_test, y_test, feature_names


def print_classify_results_accuracy(train_accuracy: float, test_accuracy: float) -> None:
    print(f"{Fore.LIGHTMAGENTA_EX}Training accuracy: {Fore.WHITE}", train_accuracy)
    print(f"{Fore.LIGHTMAGENTA_EX}Testing accuracy: {Fore.WHITE}", test_accuracy)


def print_classify_results_error(train_error: float, test_error: float) -> None:
    print(f"{Fore.LIGHTMAGENTA_EX}Training Error: {Fore.WHITE}", train_error)
    print(f"{Fore.LIGHTMAGENTA_EX}Testing Error: {Fore.WHITE}", test_error)


def accuracy_to_error(accuracy: float) -> float:
    """
    Converts an accuracy score to an error score.

    Parameters:
    - accuracy (float): The accuracy score to convert.

    Returns:
    - float: The error score.
    """

    return 1 - accuracy


def q1(
    X_train: DataFrame,
    y_train: DataFrame,
    X_test: DataFrame,
    y_test: DataFrame,
    feature_names: List[str],
) -> None:
    """
    Question 1
    Fit a single Decision Tree classifier to the training data
    with default parameters and calculate E_{train} and
    E_{hold-out}

    Parameters:
    - X_train (DataFrame): The training data features.
    - y_train (DataFrame): The training data target variable.
    - X_test (DataFrame): The testing data features.
    - y_test (DataFrame): The testing data target variable.
    - feature_names (list[str]): The names of the features.

    Returns:
    - None
    """

    max_tree_depth = 3

    decision_tree_model = decision_tree.DTClassifier(X_train, y_train, X_test, y_test, feature_names, max_tree_depth=max_tree_depth)

    test_preds, train_accuracy, test_accuracy = decision_tree_model.classify()

    train_error = accuracy_to_error(train_accuracy)
    test_error = accuracy_to_error(test_accuracy)

    print_classify_results_error(train_error, test_error)

    # Plot decision boundaries
    decision_tree_model.plot_decision_regions(resolution=0.02)


def q2(
    X_train: DataFrame,
    y_train: DataFrame,
    X_test: DataFrame,
    y_test: DataFrame,
    feature_names: List[str],
) -> None:
    """
    Question 2
    Fit a bagging ensemble method to the same training data using the Random

    Forest algorighm. Calculate E_{train} and E_{hold-out} for the ensemble.

    Parameters:
    - X_train (DataFrame): The training data features.
    - y_train (DataFrame): The training data target variable.
    - X_test (DataFrame): The testing data features.
    - y_test (DataFrame): The testing data target variable.
    - feature_names (list[str]): The names of the features.

    Returns:
    - None
    """

    n_trees = 100

    random_forest_model = random_forest.RFClassifier(X_train, y_train, X_test, y_test, feature_names, n_trees=n_trees)

    test_preds, train_accuracy, test_accuracy = random_forest_model.classify()

    train_error = accuracy_to_error(train_accuracy)
    test_error = accuracy_to_error(test_accuracy)

    print_classify_results_error(train_error, test_error)

    # Plot decision boundaries
    random_forest_model.plot_decision_regions(resolution=0.02)


def q3(
    X_train: DataFrame,
    y_train: DataFrame,
    X_test: DataFrame,
    y_test: DataFrame,
    feature_names: List[str],
) -> None:
    """
    Question 3
    Compare the results of the Decision Tree and Random Forest classifiers

    Parameters:
    - X_train (DataFrame): The training data features.
    - y_train (DataFrame): The training data target variable.
    - X_test (DataFrame): The testing data features.
    - y_test (DataFrame): The testing data target variable.
    - feature_names (list[str]): The names of the features.

    Returns:
    - None
    """

    n_trees_iterate = range(1, 100, 1)
    max_tree_depth_iterate = range(4, 5, 1)

    results = []

    for n_trees in n_trees_iterate:
        for max_tree_depth in max_tree_depth_iterate:
            random_forest_model = random_forest.RFClassifier(
                X_train, y_train, X_test, y_test, feature_names, n_trees=n_trees, max_tree_depth=max_tree_depth
            )
            test_preds, train_accuracy, test_accuracy = random_forest_model.classify()

            print(f"{Fore.LIGHTMAGENTA_EX}Random Forest with {n_trees} trees and max depth {max_tree_depth}")
            print_classify_results_accuracy(train_accuracy, test_accuracy)

            result = (n_trees, max_tree_depth, train_accuracy, test_accuracy)

            results.append(result)

    # Plot the results on a graph with x-axis as n_trees and y_axis as accuracy
    # Plot a curve for each max_tree_depth

    depth_results = {}
    for result in results:
        n_trees, max_tree_depth, train_accuracy, test_accuracy = result
        if max_tree_depth not in depth_results:
            depth_results[max_tree_depth] = {"n_trees": [], "train_accuracy": [], "test_accuracy": []}
        depth_results[max_tree_depth]["n_trees"].append(n_trees)
        depth_results[max_tree_depth]["train_accuracy"].append(train_accuracy)
        depth_results[max_tree_depth]["test_accuracy"].append(test_accuracy)

    # Plot the results
    for max_tree_depth, data in depth_results.items():
        plt.plot(data["n_trees"], data["test_accuracy"], label=f"Max Depth {max_tree_depth}")

    plt.xlabel("Number of Trees")
    plt.ylabel("Test Accuracy")
    plt.title("Random Forest Classifier Performance")
    plt.legend()
    plt.show()


def main():
    current_folder = os.path.dirname(__file__)
    data_folder = os.path.join(current_folder, "data")

    num_questions = 3
    questions = range(1, num_questions + 1)

    # Load Data
    data, X_train, y_train, X_test, y_test, feature_names = process_classification_data(data_folder, randomise=True)

    if len(sys.argv) > 1:
        if sys.argv[1] == "1":
            q1(X_train, y_train, X_test, y_test, feature_names)
        elif sys.argv[1] == "2":
            q2(X_train, y_train, X_test, y_test, feature_names)
        elif sys.argv[1] == "3":
            q3(X_train, y_train, X_test, y_test, feature_names)
        else:
            print(f"{Fore.RED}Invalid question number{Style.RESET_ALL}\n.Questions: {', '.join(questions)}")

        sys.exit(0)
    # Run all questions

    # Question 1 - Decision Tree
    q1(X_train, y_train, X_test, y_test, feature_names)

    # Question 2 - Random Forest
    q2(X_train, y_train, X_test, y_test, feature_names)

    # Question 3 - Comparison of Decision Tree and Random Forest
    q3(X_train, y_train, X_test, y_test, feature_names)


if __name__ == "__main__":
    main()
