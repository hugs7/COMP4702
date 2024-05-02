"""
Week 10 Practical - Ensemble Methods
Developer: Hugo Burton
Date: 02/05/2024
"""

import os
from typing import List, Union
from pandas import DataFrame
from colorama import Fore, Style

import load_data
import decision_tree
import random_forest

def process_classification_data(
    data_folder, randomise: bool
) -> Union[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, list[str]]:
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


def q1(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, y_test: DataFrame, feature_names: List[str]) -> None:
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

    decision_tree_model = decision_tree.DTClassifier(
        X_train, y_train, X_test, y_test, feature_names, max_tree_depth=max_tree_depth
    decision_tree_model = decision_tree.DTClassifier(
        X_train, y_train, X_test, y_test, feature_names
    )

    test_preds, train_accuracy, test_accuracy = decision_tree_model.classify()

    train_error = accuracy_to_error(train_accuracy)
    test_error = accuracy_to_error(test_accuracy)

    print_classify_results_error(train_error, test_error)

    # Plot decision boundaries
    decision_tree_model.plot_decision_regions(test_preds, resolution=0.02)

    
def q2(X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, y_test: DataFrame, feature_names: List[str]) -> None:
    """
    Question 2
    Fit a bagging ensemble method to the same training data using the Random 
    Forest algorighm. Calculate E_{train} and E_{hold-out} for the ensemble.
    """

    n_trees = 100

    random_forest_model = random_forest.RFClassifier(
        X_train, y_train, X_test, y_test, feature_names, n_trees=n_trees
    )

    test_preds, train_accuracy, test_accuracy = random_forest_model.classify()

    train_error = accuracy_to_error(train_accuracy)
    test_error = accuracy_to_error(test_accuracy)

    print_classify_results_error(train_error, test_error)

    # Plot decision boundaries
    random_forest_model.plot_decision_regions(test_preds, resolution=0.02)

def main():
    current_folder = os.path.dirname(__file__)
    data_folder = os.path.join(current_folder, "data")

    # Load Data
    data, X_train, y_train, X_test, y_test, feature_names = process_classification_data(data_folder, randomise=True)

    # Question 1 - Decision Tree
    q1(X_train, y_train, X_test, y_test, feature_names)

    # Question 2 - Random Forest
    q2(X_train, y_train, X_test, y_test, feature_names)





if __name__ == "__main__":
    main()