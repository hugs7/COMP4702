"""
Week 3 Practical
Hugo Burton
07/03/2024
"""

import os
from typing import Union
from colorama import Fore
from pandas import DataFrame

import load_data
import scatterplot
import knn
import decision_tree
import sys


CLASSIFY = "classify"
REGRESS = "regress"

KNN = "knn"
TREE = "tree"


def process_classification_data(
    data_folder, show_scatterplot: bool = False
) -> Union[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, list[str]]:
    # Load the data
    file_path = os.path.join(data_folder, "w3classif.csv")

    data = load_data.load_data(file_path)

    data = load_data.tag_data(data, ["X1", "X2", "Y"])

    if show_scatterplot:
        # Show the data as scatterplot
        scatterplot.scatterplot_with_colour(data, "X1", "X2", "Y")

    # Randomise the data
    data_randomised = load_data.shuffle_data(data)

    feature_names = ["X1", "X2"]
    # Split the data into training and testing data
    X = data_randomised.loc[:, feature_names]
    y = data_randomised.loc[:, "Y"]
    ratio = 0.3

    X_train, y_train, X_test, y_test = load_data.test_train_split(X, y, ratio=ratio)

    return data, X_train, y_train, X_test, y_test, feature_names


def print_classify_results(train_accuracy: float, test_accuracy: float) -> None:
    print(f"{Fore.LIGHTMAGENTA_EX}Training accuracy: {Fore.WHITE}", train_accuracy)
    print(f"{Fore.LIGHTMAGENTA_EX}Testing accuracy: {Fore.WHITE}", test_accuracy)
    print("")
    print(f"{Fore.LIGHTMAGENTA_EX}Training MCR: {Fore.WHITE}", 1 - train_accuracy)
    print(f"{Fore.LIGHTMAGENTA_EX}Testing MCR: {Fore.WHITE}", 1 - test_accuracy)


def knn_classify(data_folder):
    """
    Classification task using knn
    """

    data, X_train, y_train, X_test, y_test, feature_names = process_classification_data(
        data_folder, show_scatterplot=True
    )

    # Apply the knn classifier
    knn_classifier = knn.KNNClassify(
        X_train, y_train, X_test, y_test, feature_names, k=3
    )

    test_preds, train_accuracy, test_accuracy = knn_classifier.classify()

    print_classify_results(train_accuracy, test_accuracy)

    # Plot decision regions
    knn_plot_title = f"k-NN decision regions (k = {knn_classifier.get_k()})"
    knn_classifier.plot_decision_regions(
        test_preds, resolution=0.02, plot_title=knn_plot_title
    )


def knn_regress(data_folder):
    """
    Regression task
    """

    file_path = os.path.join(data_folder, "w3regr.csv")

    data = load_data.load_data(file_path)

    data = load_data.tag_data(data, ["X", "Y"])
    print(f"{Fore.RED} Data Loaded {Fore.WHITE}")
    # Plot the data
    scatterplot.scatterplot_2d(data)

    feature_names = ["X"]

    X = data.loc[:, feature_names]
    y = data.loc[:, "Y"]

    X_train, X_test, y_train, y_test = load_data.test_train_split(X, y)

    # Create model
    knn_regressor = knn.KNNRegress(X_train, X_test, y_train, y_test, feature_names, k=3)

    # Apply the knn regressor
    test_preds, train_accuracy, test_accuracy = knn_regressor.regress()

    print(f"{Fore.LIGHTMAGENTA_EX}Training R2: {Fore.WHITE}", train_accuracy)
    print(f"{Fore.LIGHTMAGENTA_EX}Testing R2: {Fore.WHITE}", test_accuracy)

    # Plot the regression line
    knn_regressor.plot_regression_line(test_preds)


def decision_tree_classify(data_folder):
    """
    Classification task using decision tree
    """

    data, X_train, y_train, X_test, y_test, feature_names = process_classification_data(
        data_folder, show_scatterplot=False
    )

    # Create the decision tree classifier

    decision_tree_model = decision_tree.DecisionTree(
        X_train, y_train, X_test, y_test, feature_names
    )

    test_preds, train_accuracy, test_accuracy = decision_tree_model.classify()

    print_classify_results(train_accuracy, test_accuracy)

    # Plot decision tree regions
    decision_tree_model.plot_decision_regions(X_test, test_preds, resolution=0.02)


def decision_tree_regress(data_folder):
    raise NotImplementedError("Decision tree regression is not implemented yet.")


def print_invalid_task_type():
    print(f"Invalid task type. Please choose '{CLASSIFY}' or '{REGRESS}'.")
    sys.exit(1)


def main():
    current_folder = os.path.dirname(__file__)
    data_folder = os.path.join(current_folder, "data")

    if len(sys.argv) != 3:
        msg = "Usage: python w3prac.py <classifier> <type>"
        print(msg)
        sys.exit(1)

    classifier = sys.argv[1]
    task_type = sys.argv[2]

    if classifier == KNN:
        if task_type == CLASSIFY:
            knn_classify(data_folder)
        elif task_type == REGRESS:
            knn_regress(data_folder)
        else:
            print_invalid_task_type()
    elif classifier == TREE:
        if task_type == CLASSIFY:
            decision_tree_classify(data_folder)
        elif task_type == REGRESS:
            decision_tree_regress(data_folder)
        else:
            print_invalid_task_type()
    else:
        print(f"Invalid classifier. Please choose '{KNN}' or '{TREE}'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
